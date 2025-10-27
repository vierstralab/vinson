import torch
import copy

import lightning as L

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
)
from torchmetrics.regression import PearsonCorrCoef
from torch.nn import BCEWithLogitsLoss

from vinson.loss import (
    PoissonNLL,
    binomial_mixture_normed_loss,
)


class CellEmbedding(torch.nn.Module):
    """
    This a simple multilayer perceptron to encode cell states from an embedding
    'n_inputs' is the dimension of the embedding space.
    """

    def __init__(self, n_inputs, n_nodes=1024, n_outputs=128, n_layers=0):
        super(CellEmbedding, self).__init__()

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_outputs = n_outputs
        self.n_layers = n_layers

        self.ifc = torch.nn.Linear(n_inputs, n_nodes)
        self.irelu = torch.nn.ReLU()

        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for i in range(self.n_layers)]
        )
        self.relus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(self.n_layers)]
        )

        self.ffc = torch.nn.Linear(n_nodes, n_outputs)

    def forward(self, embed):
        x = self.irelu(self.ifc(embed))
        for i in range(self.n_layers):
            x = self.relus[i](self.fcs[i](x))
        x = self.ffc(x)
        return x


class BassetTrunk(torch.nn.Module):
    def __init__(self):
        super(BassetTrunk, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=4, out_channels=300, kernel_size=19, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=300, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, padding=(3 - 1) // 2),
        )
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=300, out_channels=200, kernel_size=11, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=200, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, padding=(4 - 1) // 2),
        )
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=200, out_channels=200, kernel_size=7, padding="same"
            ),
            torch.nn.BatchNorm1d(num_features=200, momentum=0.1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=4, padding=(4 - 1) // 2),
        )
        self.relu3 = torch.nn.ReLU()

        # self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = self.relu3(x)

        # flatten
        # x = self.flatten(x)
        x = torch.flatten(x, start_dim=1)

        return x


class BassetTrunkEmbed(BassetTrunk):
    def __init__(self, n_embed_outputs):
        super(BassetTrunkEmbed, self).__init__()

        # TODO: inference output size from BassetTrunk convolutional layers
        self.bias2 = torch.nn.Linear(n_embed_outputs, self.layer2[0].out_channels)
        self.bias3 = torch.nn.Linear(n_embed_outputs, self.layer3[0].out_channels)

    def forward(self, x, embed):
        x = self.layer1(x)
        x = self.relu1(x)

        x_conv = self.layer2(x)
        x_bias = self.bias2(embed).unsqueeze(-1)
        x = self.relu2(x_conv + x_bias)

        x_conv = self.layer3(x)
        x_bias = self.bias3(embed).unsqueeze(-1)
        x = self.relu3(x_conv + x_bias)

        # Flatten features
        x = torch.flatten(x, start_dim=1)

        return x


class AbstractBaseSequenceModel(L.LightningModule):
    def __init__(
        self,
        trunk_model,
        seqlen=1344,
        optimizer=None,
        lr_scheduler=None,
        optimizer_kwargs=dict(),
        lr_scheduler_kwargs=dict(),
    ):
        super().__init__()

        self.trunk = trunk_model
        self.seqlen = seqlen

        # Common architecture
        self.fc1 = torch.nn.LazyLinear(1024)
        self.bn1 = torch.nn.BatchNorm1d(1024, momentum=0.1)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.relu1 = torch.nn.ReLU()

        self.fc2 = torch.nn.LazyLinear(1024)
        self.bn2 = torch.nn.BatchNorm1d(1024, momentum=0.1)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.relu2 = torch.nn.ReLU()

        self.final = torch.nn.LazyLinear(1)

        # Optimizer setup
        self.optimizer = optimizer or torch.optim.AdamW
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.train_metrics = MetricCollection({}, prefix="train_")
        self.valid_metrics = MetricCollection({}, prefix="val_")

    def init_metrics(self):
        raise NotImplementedError("Subclasses of AbstractBaseSequenceModel must implement init_metrics method.")

    def forward_fc(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu2(x)

        return x

    def forward_final(self, x):
        return self.final(x)

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        """ """

        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)

        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }


class BaseSequenceModel(AbstractBaseSequenceModel):
    def __init__(
        self,
        trunk_model,
        seqlen=1344,
        regression=False,
        optimizer=None,
        lr_scheduler=None,
        optimizer_kwargs=dict(),
        lr_scheduler_kwargs=dict(),
    ):
        super().__init__(
            trunk_model,
            seqlen,
            optimizer,
            lr_scheduler,
            optimizer_kwargs,
            lr_scheduler_kwargs,
        )
        self.regression = regression

        self.loss = (
            PoissonNLL(reduction="none")
            if self.regression
            else BCEWithLogitsLoss(reduction="none")
        )
        
        self.init_metrics()


    def init_metrics(self):
        if self.regression:
            self.train_metrics = MetricCollection(
                {
                    "pcc": PearsonCorrCoef(),
                },
                prefix="train_",
            )
        else:
            self.train_metrics = MetricCollection(
                {
                    "auroc": BinaryAUROC(),
                    "aupr": BinaryAveragePrecision(),
                    "mcc": BinaryMatthewsCorrCoef(),
                },
                prefix="train_",
            )

        self.valid_metrics = self.train_metrics.clone(prefix="val_")

    def init_model(self):
        self(torch.zeros((2, 4, self.seqlen)))
        return self

    def forward(self, seq, exp=False):
        features = self.trunk(seq)

        x = self.forward_fc(features)
        x = self.forward_final(x)

        if exp:
            x = self.exp(x)

        return x

    def _forward_from_batch(self, batch):
        X_seq = batch["ohe_seq"]
        y = self(X_seq).squeeze()
        return y

    def _run_step_regression(self, y, read_depth, bg, density):
        pseudocount = torch.tensor(1e-6, device=self.device)
        
        log_pred_counts = torch.logaddexp(
            y - torch.tensor(1e6, device=self.device).log() + torch.log(read_depth), torch.log(bg + pseudocount)
        )
        target_counts = (density / 1e6 * read_depth) + pseudocount

        return log_pred_counts, target_counts # y_hat, y

    def _run_step_classification(self, y, indicator):
        return y, indicator.float()  # y_hat, y

    def _run_step(self, batch):
        """
        Internal step function to parse batch and run forward + step

        Returns:
        y: model predictions
        target: ground truth values
        """
        weight = batch["weight"]

        y = self._forward_from_batch(batch)
        if self.regression:
            return self._run_step_regression(
                y,
                read_depth=batch["read_depth"],
                bg=batch["bg"],
                density=batch["density"]
            ), weight
        else:
            return self._run_step_classification(y, batch["class"] == 1), weight

    def step(self, batch, batch_idx):
        (y_hat, y), weight = self._run_step(batch)

        loss = self.loss(y_hat, y)
        loss *= weight
        loss = loss.mean()

        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, *_ = self.step(batch, batch_idx)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch, batch_idx)

        if self.regression:
            self.valid_metrics.update(
                torch.logaddexp(
                    y_hat, 
                    torch.tensor(1, device=self.device).log()
                ),
                (y + 1).log(),
            )
        else:
            self.valid_metrics.update(torch.sigmoid(y_hat), y.int())
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)

        if self.lr_scheduler is None:
            return optimizer

        scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "lr",
            },
        }


class EmbedModel(BaseSequenceModel):
    """Sequence with embeddings model"""

    def __init__(self, trunk: BassetTrunkEmbed, embed: CellEmbedding, *args, **kwargs):
        super().__init__(trunk, *args, **kwargs)

        self.embedding = embed

        self.save_hyperparameters(ignore=["trunk", "embed"])

    def init_model(self):
        self(
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, self.embedding.n_inputs)),
        )

    def forward(self, seq, embed):
        x = self.embedding(embed)
        x = self.trunk(seq, x)

        x = self.forward_fc(x)
        x = self.forward_final(x)

        return x

    def _forward_from_batch(self, batch):
        X_seq = batch["ohe_seq"]
        X_embed = batch["embed"]
        y = self(X_seq, X_embed).squeeze()
        return y

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
       return super().validation_step(batch, batch_idx)


class VariantEmbedModel(AbstractBaseSequenceModel):
    def __init__(self, trunk, embed, *args, **kwargs):
        super().__init__(trunk, *args, **kwargs)
    
        self.loss = binomial_mixture_normed_loss
        self.embedding = embed
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

    def forward(self, seq_ref, seq_alt, embed):
        x = self.embedding(embed)
        ref_features = self.trunk(seq_ref, x)
        alt_features = self.trunk(seq_alt, x)

        x = torch.subtract(ref_features, alt_features)

        x = self.forward_fc(x)
        x = self.forward_final(x)
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

        loss = self.loss(
            y, 
            ref_counts=ref_counts,
            total_counts=total_counts,
            bad_score=bad_score,
            reduction="none"
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

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss


class VariantEmbedModelWrapper(L.LightningModule):
    """Wrapper class for VariantModel to perform only inference"""

    def __init__(self, model: "VariantEmbedModel"):
        super().__init__()
        self.model = model

        # Make independent ref/alt branches
        self.embedding_ref = copy.deepcopy(model.embedding)
        self.embedding_alt = copy.deepcopy(model.embedding)
        self.trunk_ref = copy.deepcopy(model.trunk)
        self.trunk_alt = copy.deepcopy(model.trunk)
        for mod in [
            self.trunk_ref, self.trunk_alt, 
            self.embedding_ref, self.embedding_alt
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
