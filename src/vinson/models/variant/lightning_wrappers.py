from typing import Any, Dict, Optional
import copy

import torch

from torchmetrics import MetricCollection
from torchmetrics.regression import PearsonCorrCoef

import lightning as L

from vinson.models.sequence.lightning_wrappers import (
    AbstractSequenceModel,
    SequenceEmbedModel,
)

from vinson.optim.loss import PoissonNLLLoss, BinomialMixtureNLLLoss

from vinson.models.shared import MLPBlock, initialize_weights


class VariantEmbedModel(AbstractSequenceModel):
    def __init__(
        self,
        trunk_model: torch.nn.Module,
        head_model: MLPBlock,
        lr_scheduler: Optional[str] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        init_weights: bool = True,
    ):
        super().__init__(
            trunk_model=trunk_model,
            head_model=head_model,
            lr_scheduler=lr_scheduler,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            init_weights=init_weights,
        )

        self.criterion = BinomialMixtureNLLLoss(relative=True, reduction="none")
        self.init_metrics()

        self.save_hyperparameters(ignore=["trunk_model", "head_model"])

    def init_metrics(self):
        self.train_metrics = MetricCollection(
            {
                "pcc": PearsonCorrCoef(),
            },
            prefix="train_",
        )

        self.valid_metrics = self.train_metrics.clone(prefix="val_")

    def forward_final(self, x):
        x = self.final(x)
        return x

    def forward(self, seq_ref, seq_alt, embed):
        # x = self.embed_model(embed)
        features_ref = self.trunk_model(seq_ref, embed)
        features_alt = self.trunk_model(seq_alt, embed)

        # x = torch.subtract(ref_features, alt_features)
        x = torch.cat([features_ref, features_alt], dim=-1)

        x = self.head_model(x)
        x = self.forward_final(
            x
        )  # in variant model, outputs are always logits of ES -infinity to +infinity
        return x

    def _forward_from_batch(self, batch):
        X_ref = batch["ohe_seq_ref"]
        X_alt = batch["ohe_seq_alt"]
        X_embed = batch["embed"]
        y = self(X_ref, X_alt, X_embed).squeeze()
        return y

    def step(self, batch, batch_idx):
        y = self._forward_from_batch(batch)

        ref_counts, total_counts, bad_score = (
            batch["ref_counts"],
            batch["total_counts"],
            batch["bad_score"],
        )

        loss = self.criterion(
            y,
            ref_counts,
            total_counts,
            bad_score,
        )

        loss *= batch["weight"]
        loss = loss.mean()

        return loss, y, (ref_counts, total_counts, bad_score)

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        lfc = batch["lfc"]

        self.valid_metrics.update(y_hat, lfc)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch, batch_idx: int = None) -> torch.Tensor:
        return self._forward_from_batch(batch)

    def freeze_trunk(self):
        for param in self.trunk_model.parameters():
            param.requires_grad = False
        print("Trunk frozen.")

    def unfreeze_trunk(self):
        for param in self.trunk_model.parameters():
            param.requires_grad = True
        print("Trunk unfrozen.")

    @classmethod
    def from_sequence_embed_model(
        cls, sequence_embed_model: SequenceEmbedModel, head_model: MLPBlock, **kwargs
    ):
        model = cls(
            trunk_model=sequence_embed_model.trunk_model,
            head_model=head_model,
            init_weights=False,
            **kwargs,
        )
        model.head_model.apply(initialize_weights)
        return model


class VariantEmbedModelWrapper(L.LightningModule):
    """Wrapper class for VariantModel to perform only inference"""

    def __init__(self, model: VariantEmbedModel):
        super().__init__()
        self.model = model

        self.trunk_ref = copy.deepcopy(model.trunk_model)
        self.trunk_alt = copy.deepcopy(model.trunk_model)

        for mod in [
            self.trunk_ref,
            self.trunk_alt,
        ]:
            mod.eval()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # resolves 'model' from _modules
        except AttributeError:
            return getattr(super().__getattr__("model"), name)

    def forward(self, seq_ref, seq_alt, embed: torch.Tensor) -> torch.Tensor:
        """ """

        features_ref = self.trunk_ref(seq_ref, embed)
        features_alt = self.trunk_alt(seq_alt, embed.clone())

        x = torch.cat([features_ref, features_alt], dim=-1)

        x = self.model.head_model(x)
        x = self.model.forward_final(x)

        return x

    def predict_step(self, *args, **kwargs):
        return self.model.predict_step(*args, **kwargs)


class VariantEmbedDensityModel(SequenceEmbedModel):
    def __init__(
        self,
        trunk_model,
        head_model_density,
        head_model_lfc,
        lr_scheduler=None,
        optimizer_kwargs=None,
        lr_scheduler_kwargs=None,
        init_weights=True,
        loss_weight_density=1.0
    ):
        super().__init__(
            trunk_model,
            head_model_density,
            lr_scheduler,
            optimizer_kwargs,
            lr_scheduler_kwargs,
            init_weights,
            regression=True,
        )

        self.head_model_density = self.head_model
        self.final_density = self.final

        self.head_model_lfc = head_model_lfc
        self.final_lfc = torch.nn.Linear(self.head_model_lfc.output_dim, 1)
        if init_weights:
            self.head_model_lfc.apply(initialize_weights)
            self.final_lfc.apply(initialize_weights)

        self.criterion_density = self.criterion
        self.criterion_lfc = BinomialMixtureNLLLoss(relative=True, reduction="none")

        self.loss_weight_density = loss_weight_density

    def freeze_trunk(self):
        for param in self.trunk_model.parameters():
            param.requires_grad = False
        print("Trunk frozen.")

    def unfreeze_trunk(self):
        for param in self.trunk_model.parameters():
            param.requires_grad = True
        print("Trunk unfrozen.")

    def forward_final_density(self, x):
        x = self.final_density(x)
        x = torch.nn.Softplus()(x)
        return x

    def forward_final_lfc(self, x):
        x = self.final_lfc(x)
        return x

    def forward(self, seq_ref, seq_alt, embed):

        features_ref = self.trunk_model(seq_ref, embed)
        features_alt = self.trunk_model(seq_alt, embed)

        density_ref = self.head_model_density(features_ref)
        density_ref = self.forward_final_density(density_ref)

        density_alt = self.head_model_density(features_alt)
        density_alt = self.forward_final_density(density_alt)

        lfc = torch.cat([features_ref, features_alt], dim=-1)
        lfc = self.head_model_lfc(lfc)
        lfc = self.forward_final_lfc(lfc)

        return density_ref, density_alt, lfc

    def _forward_from_batch(self, batch):
        X_ref = batch["ohe_seq_ref"]
        X_alt = batch["ohe_seq_alt"]
        X_embed = batch["embed"]
        density_ref, density_alt, lfc = self(X_ref, X_alt, X_embed)
        return density_ref.squeeze(-1), density_alt.squeeze(-1), lfc.squeeze(-1)

    def step(self, batch, batch_idx):
        y_density_ref, y_density_alt, y_lfc = self._forward_from_batch(batch)
        y_density = (y_density_ref + y_density_alt) / 2

        # loss of density
        y_counts, counts = self._run_step_regression(
            y_density,
            read_depth=batch["read_depth"],
            bg=batch["bg"],
            density=batch["density"],
        )

        loss_density = self.criterion_density(y_counts, counts)

        # loss on LFC
        ref_counts, total_counts, bad_score = (
            batch["ref_counts"],
            batch["total_counts"],
            batch["bad_score"],
        )

        loss_lfc = self.criterion_lfc(y_lfc, ref_counts, total_counts, bad_score)

        # total loss
        loss = self.loss_weight_density * loss_density + loss_lfc
        loss = loss.mean()

        return (
            loss,
            y_density,
            (y_counts, counts),
            y_lfc,
            (ref_counts, total_counts, bad_score),
        )

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss, _, counts, y_lfc, _ = self.step(batch, batch_idx)

        lfc = batch["lfc"]

        # TODO: add metrics
        # counts is tuple with DHS density counts (pred, target)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss