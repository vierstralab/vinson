from typing import Any, Dict, Optional
import copy
import csv

import torch

from torchmetrics import MetricCollection
from torchmetrics.regression import PearsonCorrCoef

import lightning as L

from vinson.models.sequence.lightning_wrappers import AbstractSequenceModel, SequenceEmbedModel

from vinson.optim.loss import BinomialMixtureNLLLoss

from vinson.models.shared import MLPBlock, initialize_weights


class VariantEmbedModel(AbstractSequenceModel):
    def __init__(
        self,
        trunk_model: torch.nn.Module,
        head_model: MLPBlock,
        embed_model: MLPBlock,
        lr_scheduler: Optional[str]=None,
        optimizer_kwargs: Optional[Dict[str, Any]]=None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]]=None,
        init_weights: bool=True,
    ):
        super().__init__(
            trunk_model=trunk_model,
            head_model=head_model,
            lr_scheduler=lr_scheduler,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            init_weights=init_weights,
        )
        self.embed_model = embed_model
        if init_weights:
            self.embed_model.apply(initialize_weights)

        self.criterion = BinomialMixtureNLLLoss(relative=True, reduction="none")
        self.init_metrics()
        self.save_hyperparameters(ignore=["trunk_model", "head_model", "embed_model"])

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
        x = self.embed_model(embed)
        ref_features = self.trunk_model(seq_ref, x)
        alt_features = self.trunk_model(seq_alt, x)

        x = torch.subtract(ref_features, alt_features)

        x = self.head_model(x)
        x = self.forward_final(x) # in variant model, outputs are always logits of ES -infinity to +infinity
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

    # def training_step(self, batch, batch_idx):
    #     loss, *_ = self.step(batch, batch_idx)

    #     self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)

    #     return loss
    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)
    
        if getattr(self, "debug", False) and hasattr(self, "batch_log_file"):
            lfc = batch.get("lfc")
            ref_counts = batch.get("ref_counts")
            total_counts = batch.get("total_counts")
            bad_score = batch.get("bad_score")
    
            with open(self.batch_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    batch_idx,
                    loss.item(),
                    lfc.min().item() if lfc is not None else "",
                    lfc.max().item() if lfc is not None else "",
                    ref_counts.min().item() if ref_counts is not None else "",
                    ref_counts.max().item() if ref_counts is not None else "",
                    total_counts.min().item() if total_counts is not None else "",
                    total_counts.max().item() if total_counts is not None else "",
                    bad_score.min().item() if bad_score is not None else "",
                    bad_score.max().item() if bad_score is not None else "",
                ])
    
        self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        lfc = batch["lfc"]

        self.valid_metrics.update(y_hat, lfc)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    @classmethod
    def from_sequence_embed_model(
        cls,
        sequence_embed_model: SequenceEmbedModel,
        head_model: MLPBlock,
        **kwargs
    ):
        model = cls(
            trunk_model=sequence_embed_model.trunk_model,
            embed_model=sequence_embed_model.embed_model,
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

        # Make independent ref/alt branches
        self.embedding_ref = copy.deepcopy(model.embed_model)
        self.embedding_alt = copy.deepcopy(model.embed_model)

        self.trunk_ref = copy.deepcopy(model.trunk_model)
        self.trunk_alt = copy.deepcopy(model.trunk_model)

        for mod in [
            self.trunk_ref,
            self.trunk_alt,
            self.embedding_ref,
            self.embedding_alt,
        ]:
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False

    def __getattr__(self, name):
        if name != "model":
            try:
                return getattr(self.model, name)
            except AttributeError:
                pass
        raise AttributeError(f"{self.model} has no attribute {name}")

    def forward(self, seq_ref, seq_alt, embed: torch.Tensor) -> torch.Tensor:
        """ """
        features_ref = self.trunk_ref(seq_ref, self.embedding_ref(embed))
        features_alt = self.trunk_alt(seq_alt, self.embedding_alt(embed.clone()))

        x = torch.subtract(features_ref, features_alt)

        x = self.model.head_model(x)
        x = self.model.forward_final(x)

        return x
