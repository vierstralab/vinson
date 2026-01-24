import torch
from typing import Any, Dict, Optional, Tuple, Union, List

import lightning as L

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
)
from torchmetrics.regression import PearsonCorrCoef

from torch.nn import BCEWithLogitsLoss

from vinson.optim.loss import PoissonNLLLoss

from .cell_classifier import EmbeddingMLP




class CellEmbedding(EmbeddingMLP):
    """
    This a simple multilayer perceptron to encode cell states from an embedding
    'n_inputs' is the dimension of the embedding space.
    """

    def __init__(
        self,
        n_inputs: int,
        n_nodes: int = 1024,
        n_outputs: int = 128,
        n_layers: int = 0,
        activations: Union[List[str], str] = "relu",
    ) -> None:
        super().__init__(
            n_inputs=n_inputs,
            n_nodes=n_nodes,
            n_layers=n_layers,
            activations=activations,
        )
        self.n_outputs = n_outputs

        self.ffc = torch.nn.Linear(n_nodes, self.n_outputs)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        x = super().forward(embed)
        x = self.ffc(x)
        return x


class BassetTrunk(torch.nn.Module):
    def __init__(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, n_embed_outputs: int) -> None:
        super().__init__()

        self.bias2 = torch.nn.Linear(n_embed_outputs, self.layer2[0].out_channels)
        self.bias3 = torch.nn.Linear(n_embed_outputs, self.layer3[0].out_channels)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
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
        trunk_model: torch.nn.Module,
        seqlen: int = 1344,
        lr_scheduler: Optional[str] = None,
        optimizer_kwargs: Dict[str, Any] = dict(),
        lr_scheduler_kwargs: Dict[str, Any] = dict(),
    ) -> None:
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
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.train_metrics = MetricCollection({}, prefix="train_")
        self.valid_metrics = MetricCollection({}, prefix="val_")

    def init_metrics(self) -> None:
        raise NotImplementedError(
            "Subclasses of AbstractBaseSequenceModel must implement init_metrics method."
        )

    def forward_fc(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.relu2(x)

        return x

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        """ """
        lr_scheduler = LR_SCHEDULERS.get(self.lr_scheduler, None)
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_kwargs)

        if lr_scheduler is None:
            return optimizer

        scheduler = lr_scheduler(optimizer, **self.lr_scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        }


class BaseSequenceModel(AbstractBaseSequenceModel):
    def __init__(
        self,
        trunk_model: torch.nn.Module,
        seqlen: int = 1344,
        regression: bool = False,
        log_output: bool = True,
        lr_scheduler: Optional[str] = None,
        optimizer_kwargs: Dict[str, Any] = dict(),
        lr_scheduler_kwargs: Dict[str, Any] = dict(),
    ) -> None:
        super().__init__(
            trunk_model=trunk_model,
            seqlen=seqlen,
            lr_scheduler=lr_scheduler,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )
        self.regression = regression
        self.log_output = log_output

        self.criterion = (
            PoissonNLLLoss(log_input=log_output, reduction="none")
            if self.regression
            else BCEWithLogitsLoss(reduction="none")
        )

        self.init_metrics()

    def init_metrics(self) -> None:
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

    def init_model(self) -> "BaseSequenceModel":
        self(torch.zeros((2, 4, self.seqlen)))
        return self

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        features = self.trunk(seq)

        x = self.forward_fc(features)
        x = self.forward_final(x)

        return x

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final(x)
        if not self.log_output:
            x = torch.relu(x)
        return x

    def _forward_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        X_seq = batch["ohe_seq"]
        y = self(X_seq).squeeze()
        return y

    def _run_step_regression(
        self,
        y: torch.Tensor,
        read_depth: torch.Tensor,
        bg: torch.Tensor,
        density: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        million = torch.tensor(1e6, device=self.device)

        if self.log_output:
            input = torch.logaddexp(
                y - million.log() + torch.log(read_depth), torch.log(bg)
            )
        else:
            input = y / million * read_depth + bg

        target_counts = torch.clip(density / million * read_depth, bg, None)

        return input, target_counts  # y_hat, y

    def _run_step_classification(
        self, y: torch.Tensor, indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return y, indicator.float()  # y_hat, y

    def _run_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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
                density=batch["density"],
            ), weight
        else:
            return self._run_step_classification(y, batch["class"] == 1), weight

    def step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (y_hat, y), weight = self._run_step(batch)

        loss = self.criterion(y_hat, y)
        loss *= weight
        loss = loss.mean()

        return loss, y_hat, y

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, *_ = self.step(batch, batch_idx)

        self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, y_hat, y = self.step(batch, batch_idx)

        if self.regression:
            if not self.log_output:
                y_hat = torch.log(y_hat + 1e-6)
            self.valid_metrics.update(y_hat, y)
        else:
            self.valid_metrics.update(torch.sigmoid(y_hat), y.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()


class EmbedModel(BaseSequenceModel):
    """Sequence with embeddings model"""

    def __init__(
        self, trunk: torch.nn.Module, embed: CellEmbedding, *args, **kwargs
    ) -> None:
        super().__init__(trunk, *args, **kwargs)

        self.embedding = embed

        self.save_hyperparameters(ignore=["trunk", "embed"])

    def init_model(self) -> "EmbedModel":
        self(
            torch.zeros((2, 4, self.seqlen)),
            torch.zeros((2, self.embedding.n_inputs)),
        )
        return self

    def forward(self, seq: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        x = self.embedding(embed)
        x = self.trunk(seq, x)

        x = self.forward_fc(x)
        x = self.forward_final(x)

        return x

    def _forward_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        X_seq = batch["ohe_seq"]
        X_embed = batch["embed"]
        y = self(X_seq, X_embed).squeeze()
        return y

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return super().validation_step(batch, batch_idx)
