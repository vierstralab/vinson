import torch
import lightning as L
from typing import Dict, Any


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
    def __init__(self, n_inputs: int, n_nodes: int = 64, n_layers: int = 1, dropout: float = 0.3):
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
    >>> model = BaseCellClassifierModel(
    ...     n_inputs=1000,
    ...     output_dict={'cell_type': 10, 'disease_state': 2},
    ...     n_nodes=128,
    ...     n_layers=2
    ... )
    >>> inputs = torch.randn(32, 1000)
    >>> outputs = model(inputs)
    >>> print(outputs.keys())  # dict_keys(['cell_type', 'disease_state'])
    """

    def __init__(self, n_inputs: int, output_dict: Dict[str, int], **kwargs):
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

        self.criterion = torch.nn.CrossEntropyLoss()

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

        return loss

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-2)
        return optimizer


# class CellAndPathologicalStateClassifierModel(BaseCellClassifierModel):
#     def __init__(self, n_inputs, n_cell_categories, n_pathological_states, **kwargs):
#         super(CellAndPathologicalStateClassifierModel, self).__init__(n_inputs, n_cell_categories, **kwargs)

#         self.head_cell_category = torch.nn.Linear(self.trunk.n_nodes, n_cell_categories)
#         self.head_pathological_state = torch.nn.Linear(self.trunk.n_nodes, n_pathological_states)

#         self.criterion = torch.nn.CrossEntropyLoss()

#         self.save_hyperparameters()

#     def forward(self, x):
#         x = self.trunk(x)
#         cell_category = self.head_cell_category(x)
#         pathological_state = self.head_pathological_state(x)
#         return cell_category, pathological_state

#     def training_step(self, batch, batch_idx):
#         X, y_cell_category, y_pathological_state = (
#             batch["embed"],
#             batch["cell_category"],
#             batch["pathological_state"],
#         )

#         _y_cell_category, _y_pathological_state = self(X)

#         loss_cell_category = self.loss_fn(_y_cell_category, y_cell_category)
#         loss_pathological_state = self.loss_fn(_y_pathological_state, y_pathological_state)
#         loss = loss_cell_category + loss_pathological_state

#         self.log(
#             "loss",
#             loss,
#             on_step=True,
#             on_epoch=True,
#             sync_dist=True,
#             prog_bar=True,
#         )

#         return loss

#     def validation_step(self, batch, batch_idx):
#         X, y_cell_category, y_pathological_state = (
#             batch["embed"],
#             batch["cell_category"],
#             batch["pathological_state"],
#         )

#         _y_cell_category, _y_pathological_state = self(X)

#         loss_cell_categories = self.loss_fn(_y_cell_category, y_cell_category)
#         loss_pathological_state = self.loss_fn(_y_pathological_state, y_pathological_state)
#         loss = loss_cell_categories + loss_pathological_state

#         self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=1e-2)
#         return optimizer
