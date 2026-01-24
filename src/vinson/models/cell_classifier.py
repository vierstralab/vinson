from typing import Dict, Any, Union, List

import torch
import lightning as L

from vinson.utils.hparams import get_activations, get_hidden_dims
from vinson.utils.optim import configure_optimizer


class EmbeddingMLP(torch.nn.Module):
    """
    Multi-layer perceptron for embedding inputs with batch normalization and dropout.

    This MLP consists of a stack of linear layers defined by `hidden_dims`,
    each followed by batch normalization, configurable activation, and dropout.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    hidden_dims : int or list[int]
        Hidden layer dimensions. A scalar creates a single hidden layer.
    dropout : float, optional
        Dropout probability.
    activations : str or list[str], optional
        Activation function(s) to use. Scalars are broadcast per layer.
    """
    def __init__(
        self,
        n_inputs: int,
        hidden_dims: Union[int, List[int]],
        dropout: float = 0.3,
        activations: Union[List[str], str] = "silu",
    ):
        super().__init__()

        hidden_dims = get_hidden_dims(hidden_dims)
        n_layers = len(hidden_dims)

        dims = [n_inputs] + hidden_dims

        assert n_layers >= 1, "n_layers must be at least 1"

        self.n_inputs = n_inputs
        self.n_layers = n_layers

        
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
        )

        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(d) for d in hidden_dims]
        )

        self.activations = torch.nn.ModuleList(
            get_activations(activations, n_layers)
        )

        self.dropouts = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for _ in range(n_layers)]
        )

        assert len(self.fcs) == len(self.batch_norms) == len(self.activations) == len(self.dropouts) == n_layers

        self.output_dim = hidden_dims[-1]

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
            Output tensor of shape (batch_size, hidden_dims[-1]).
        """
        for fc, bn, act, drop in zip(
            self.fcs, self.batch_norms, self.activations, self.dropouts
        ):
            x = fc(x)
            x = bn(x)
            x = act(x)
            x = drop(x)

        return x


class CellEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedding: EmbeddingMLP, # can be any embedding
        n_outputs: int = 256,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.n_outputs = n_outputs

        self.ffc = torch.nn.Linear(self.embedding.output_dim, self.n_outputs)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        x = self.embedding(embed)
        x = self.ffc(x)
        return x


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
        n_inputs: int,
        hidden_dims: Union[int, List[int]],
        output_dict: Dict[str, int],
        lr_scheduler=None,
        optimizer_kwargs=None,
        lr_scheduler_kwargs=None,
        **kwargs,
    ):
        super().__init__()

        self.trunk = EmbeddingMLP(
            n_inputs=n_inputs,
            hidden_dims=hidden_dims,
            **kwargs
        )
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

        self.save_hyperparameters(ignore=["trunk", "heads"])


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

