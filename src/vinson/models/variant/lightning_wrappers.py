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
        ref_features = self.trunk_model(seq_ref, embed)
        alt_features = self.trunk_model(seq_alt, embed)

        # x = torch.subtract(ref_features, alt_features)
        x = torch.cat([ref_features, alt_features], dim=-1)
        
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

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)

        self.log("loss", loss, on_step=True, on_epoch=False, 
                 sync_dist=True,prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        lfc = batch["lfc"]

        self.valid_metrics.update(y_hat, lfc)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True,prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx: int=None) -> torch.Tensor:
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
        cls,
        sequence_embed_model: SequenceEmbedModel,
        head_model: MLPBlock,
        **kwargs
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
            return super().__getattr__(name)          # resolves 'model' from _modules
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
