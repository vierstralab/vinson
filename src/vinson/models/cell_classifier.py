from typing import Dict, Any

import torch
import lightning as L

from vinson.utils.optim import configure_optimizer

from vinson.models.shared import MLPBlock


## Lightning Models ##
class CellClassifierModel(L.LightningModule):
    """
    Base model for cell classification using multi-head architecture.

    This model consists of a shared trunk (MLP) followed by multiple output heads,
    each corresponding to a different classification task. It uses cross-entropy loss
    for training and is designed for multi-label or multi-task cell classification.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    output_dict : Dict[str, int]
        Dictionary mapping output head names to the number of classes for each head.
    **kwargs
        Additional keyword arguments passed to the EmbeddingMLP trunk.

    Examples
    --------
    >>> model = CellClassifierModel(
    ...     n_inputs=1000,
    ...     output_dict={'cell_type': 10, 'disease_state': 2},
    ...     hidden_dims=[128, 128],
    ...     dropout=0.2,
    ...     activations=['selu', 'silu'],
    ... )
    >>> inputs = torch.randn(32, 1000)
    >>> outputs = model(inputs)
    >>> print(outputs.keys())  # dict_keys(['cell_type', 'disease_state'])
    """
    def __init__(
        self,
        embedding: MLPBlock,
        output_dict: Dict[str, int],
        lr_scheduler=None,
        optimizer_kwargs=None,
        lr_scheduler_kwargs=None,
    ):
        super().__init__()

        self.trunk = embedding
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(
                    in_features=self.trunk.output_dim,
                    out_features=n_features
                )
                for name, n_features in output_dict.items()
            }
        )

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # Optimizer setup
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.save_hyperparameters()


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.trunk(x)
        return {k: h(x) for k, h in self.heads.items()}

    def step(self, batch: Dict[str, Any]) -> torch.Tensor:
        X = batch["embed"]
        y_hat = self(X)

        loss = 0.0
        for head_name in self.heads:
            loss += self.criterion(y_hat[head_name], batch[head_name])

        return loss.mean()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)

        self.log(
            "loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        return configure_optimizer(
            module_parameters=self.parameters(),
            optimizer_kwargs=self.optimizer_kwargs,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        )

