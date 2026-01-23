from typing import Dict, Any, Union

import torch
import lightning as L

from vinson.lr import LR_SCHEDULERS


class EmbeddingMLP(torch.nn.Module):
    """
    Multi-layer perceptron for embedding inputs with batch normalization and dropout.

    This MLP consists of an initial linear layer followed by a series of hidden layers,
    each with batch normalization, ReLU activation, and dropout. It is designed for
    feature embedding in classification tasks.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_nodes : int, optional
        Number of nodes in each hidden layer (default is 64).
    n_layers : int, optional
        Number of hidden layers (default is 1).
    dropout : float, optional
        Dropout probability (default is 0.3).
    """

    def __init__(
        self, n_inputs: int, n_nodes: int = 64, n_layers: int = 1, dropout: float = 0.3
    ):
        super(EmbeddingMLP, self).__init__()

        self.n_inputs = n_inputs
        self.n_nodes = n_nodes
        self.n_layers = n_layers

        self.ifc = torch.nn.Linear(n_inputs, n_nodes)
        self.ibn = torch.nn.BatchNorm1d(num_features=n_nodes)
        self.irelu = torch.nn.ReLU()
        self.idropout = torch.nn.Dropout(p=dropout)

        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(n_nodes, n_nodes) for i in range(self.n_layers)]
        )
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(num_features=n_nodes) for i in range(self.n_layers)]
        )
        self.relus = torch.nn.ModuleList(
            [torch.nn.ReLU() for i in range(self.n_layers)]
        )
        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for i in range(self.n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_inputs).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_nodes).
        """
        x = self.ifc(x)
        x = self.ibn(x)
        x = self.irelu(x)
        x = self.idropout(x)

        for i in range(self.n_layers):
            x = self.fcs[i](x)
            x = self.bns[i](x)
            x = self.relus[i](x)
            x = self.dropouts[i](x)

        return x


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
    ...     n_nodes=128,
    ...     n_layers=2
    ... )
    >>> inputs = torch.randn(32, 1000)
    >>> outputs = model(inputs)
    >>> print(outputs.keys())  # dict_keys(['cell_type', 'disease_state'])
    """

    def __init__(
        self,
        n_inputs: int,
        output_dict: Dict[str, int],
        lr_scheduler=None,
        optimizer_kwargs=dict(lr=5e-5, weight_decay=1e-2),
        lr_scheduler_kwargs=dict(),
        **kwargs,
    ):
        super(CellClassifierModel, self).__init__()

        self.trunk = EmbeddingMLP(n_inputs, **kwargs)
        self.heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(
                    in_features=self.trunk.n_nodes, out_features=n_features
                )
                for name, n_features in output_dict
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

        loss = torch.tensor(0)
        for head_name in self.heads.keys():
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

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
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
