from typing import Any, Dict, Optional, Tuple, Union, List

import lightning as L

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryAUROC,
)
from torchmetrics.regression import PearsonCorrCoef

import torch

from torch.nn import BCEWithLogitsLoss
import torch.nn as nn
from vinson.optim.loss import PoissonNLLLoss
from vinson.utils.optim import configure_optimizer

from vinson.models.shared import MLPBlock, initialize_weights


## Lightning Models ##
class AbstractSequenceModel(L.LightningModule):
    def __init__(
        self,
        trunk_model: torch.nn.Module,
        head_model: MLPBlock,
        n_tasks=1,
        lr_scheduler: Optional[str]=None,
        optimizer_kwargs: Optional[Dict[str, Any]]=None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]]=None,
        init_weights: bool=True,
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks

        self.trunk_model = trunk_model
        self.head_model = head_model

        self.final = torch.nn.Linear(self.head_model.output_dim, n_tasks)

        # Optimizer setup
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.train_metrics = MetricCollection({}, prefix="train_")
        self.valid_metrics = MetricCollection({}, prefix="val_")
        
        if init_weights:
            self.trunk_model.apply(initialize_weights)
            self.head_model.apply(initialize_weights)
            self.final.apply(initialize_weights)

    def init_metrics(self) -> None:
        raise NotImplementedError(
            "Subclasses of AbstractBaseSequenceModel must implement init_metrics method."
        )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.valid_metrics.compute(), sync_dist=True)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        return configure_optimizer(
            module_parameters=self.parameters(),
            optimizer_kwargs=self.optimizer_kwargs,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        )


class SequenceOnlyModel(AbstractSequenceModel):
    def __init__(
        self,
        trunk_model: torch.nn.Module,
        head_model: MLPBlock,
        regression: bool = False,
        log_output: bool = False,
        n_tasks: int = 1,
        lr_scheduler: Optional[str]=None,
        optimizer_kwargs: Optional[Dict[str, Any]]=None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]]=None,
        save_hyperparameters: bool = True,
        init_weights: bool=True,
    ) -> None:
        super().__init__(
            trunk_model=trunk_model,
            head_model=head_model,
            lr_scheduler=lr_scheduler,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            n_tasks=n_tasks,
            init_weights=init_weights,
        )
        self.regression = regression
        self.log_output = log_output

        self.criterion = (
            PoissonNLLLoss(log_input=log_output, reduction="none")
            if self.regression
            else BCEWithLogitsLoss(reduction="none")
        )
        self.init_metrics()
        if save_hyperparameters:
            self.save_hyperparameters(ignore=["trunk_model", "head_model"])

    def init_metrics(self) -> None:
        if self.regression:
            self.train_metrics = MetricCollection(
                {
                    "pcc": PearsonCorrCoef(num_outputs=self.n_tasks),
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

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        features = self.trunk_model(seq)

        x = self.head_model(features)
        x = self.forward_final(x)

        return x

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        x = self.final(x)
        if not self.log_output:
            x = torch.nn.Softplus()(x)
        return x

    def _forward_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        X_seq = batch["ohe_seq"]
        y = self(X_seq).squeeze(-1)
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

    def _run_step(self, batch: Dict[str, Any]):
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
            ), weight, y
        else:
            return self._run_step_classification(y, batch["class"] == 1), weight

    def step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (y_hat, y), weight, y_density = self._run_step(batch)

        loss = self.criterion(y_hat, y)
        if loss.ndim == 1:
            loss = loss * weight
        else:
            loss = loss * weight[:, None]
        loss = loss.mean()

        return loss, y_hat, y, y_density

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, *_ = self.step(batch, batch_idx)

        self.log("loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, y_hat, y, y_density = self.step(batch, batch_idx)

        if self.regression:
            if not self.log_output:
                y_hat = y_density
                # density = batch['density']
                density = batch["density"].squeeze(-1)
            self.valid_metrics.update(y_hat, density)
        else:
            self.valid_metrics.update(torch.sigmoid(y_hat), y.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def predict_step(self, batch, batch_idx: int=None) -> torch.Tensor:
        return self._forward_from_batch(batch)

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        out = {}

        for k, v in metrics.items():
            v = torch.as_tensor(v)
            if v.numel() > 1:
                out[k] = v.mean() # log mean over tasks for multi-task
            else:
                out[k] = v.item()

        self.log_dict(out, sync_dist=True)
        self.valid_metrics.reset()


class SequenceEmbedModel(SequenceOnlyModel):
    """Sequence with embeddings model"""

    def __init__(
            self,
            trunk_model: nn.Module,
            head_model: MLPBlock,
            lr_scheduler: Optional[str]=None,
            optimizer_kwargs: Optional[Dict[str, Any]]=None,
            lr_scheduler_kwargs: Optional[Dict[str, Any]]=None,
            init_weights: bool=True,
            **kwargs
        ):
        super().__init__(
            trunk_model=trunk_model,
            head_model=head_model,
            lr_scheduler=lr_scheduler,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            init_weights=init_weights,
            save_hyperparameters=False,
            **kwargs
        )
        self.save_hyperparameters(ignore=["trunk_model", "head_model"])

    def forward(self, seq: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        x = self.trunk_model(seq, embedding)
        x = self.head_model(x)
        x = self.forward_final(x)

        return x

    def _forward_from_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        X_seq = batch["ohe_seq"]
        X_embed = batch["embed"]
        y = self(X_seq, X_embed).squeeze(-1)
        return y

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return super().validation_step(batch, batch_idx)
