import copy

import torch
import lightning as L

from torchmetrics import MetricCollection
from torchmetrics.regression import PearsonCorrCoef

from vinson.models.sequence import AbstractBaseSequenceModel, CellEmbedding
from vinson.loss import (
    BinomialMixtureNLLLoss,
)


class VariantEmbedModel(AbstractBaseSequenceModel):
    def __init__(self, trunk: torch.nn.Module, embed: CellEmbedding, *args, **kwargs):
        super().__init__(trunk, *args, **kwargs)

        self.embedding = embed

        self.criterion = BinomialMixtureNLLLoss(relative=True, reduction="none")

        self.save_hyperparameters()

    def init_metrics(self):
        self.train_metrics = MetricCollection(
            {
                "pcc": PearsonCorrCoef(),
            },
            prefix="train_",
        )

        self.valid_metrics = self.train_metrics.clone(prefix="val_")

    def init_model(self):
        self(
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, self.embedding.n_inputs)),
        )

    def forward_final(self, x):
        x = self.final(x)
        return x

    def forward(self, seq_ref, seq_alt, embed):
        x = self.embedding(embed)
        ref_features = self.trunk(seq_ref, x)
        alt_features = self.trunk(seq_alt, x)

        x = torch.subtract(ref_features, alt_features)

        x = self.forward_fc(x)
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
            ref_counts=ref_counts,
            total_counts=total_counts,
            bad_score=bad_score,
        )

        loss *= batch["weight"]
        loss = loss.mean()

        return loss, y, (ref_counts, total_counts, bad_score)

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)

        self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)
        lfc = batch["lfc"]

        self.valid_metrics.update(y_hat, lfc)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss


class VariantEmbedModelWrapper(L.LightningModule):
    """Wrapper class for VariantModel to perform only inference"""

    def __init__(self, model: VariantEmbedModel):
        super().__init__()
        self.model = model

        # Make independent ref/alt branches
        self.embedding_ref = copy.deepcopy(model.embedding)
        self.embedding_alt = copy.deepcopy(model.embedding)

        self.trunk_ref = copy.deepcopy(model.trunk)
        self.trunk_alt = copy.deepcopy(model.trunk)

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

    def forward(self, seq_ref, seq_alt, embed):
        """ """
        features_ref = self.trunk_ref(seq_ref, self.embedding_ref(embed))
        features_alt = self.trunk_alt(seq_alt, self.embedding_alt(embed.clone()))

        x = torch.subtract(features_ref, features_alt)

        x = self.model.forward_fc(x)
        x = self.model.forward_final(x)

        return x
