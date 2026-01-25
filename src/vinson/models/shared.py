import torch
import torch.nn as nn

from typing import List, Union

from vinson.utils.hparams import get_activations, get_hidden_dims, get_batchnorms


class MLPBlock(nn.Module):
    """
    Multi-layer perceptron block with batch normalization and dropout.

    This module implements a stack of fully connected layers defined by
    `hidden_dims`, where each layer applies a linear transformation followed by dropout, activation function and batch normalization.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    hidden_dims : int or list[int]
        Output dimensions of each hidden layer. A scalar creates a single layer.
    dropout : float, optional
        Dropout probability applied after the activation in each layer.
    activations : str or list[str], optional
        Activation function(s) to use per layer. Scalars are broadcasted across layers.
    batch_norm : bool or list[bool], optional
        Whether to apply batch normalization after each layer. Scalars are broadcasted.
    batch_norm_momentum : float, optional
        Momentum parameter for batch normalization layers.
    """
    def __init__(
        self,
        n_inputs: int,
        hidden_dims: Union[int, List[int]],
        dropout: float = 0.2,
        batch_norm: Union[bool, List[bool]] = True,
        activations: Union[List[str], str] = "silu",
        batch_norm_momentum: float = 0.1,
    ):
        super().__init__()

        hidden_dims = get_hidden_dims(hidden_dims)
        n_layers = len(hidden_dims)

        dims = [n_inputs] + hidden_dims

        assert n_layers >= 1, "n_layers must be at least 1"

        self.n_inputs = n_inputs
        self.n_layers = n_layers

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
        )

        self.batch_norms = torch.nn.ModuleList(
            get_batchnorms(
                batch_norm,
                hidden_dims,
                momentum=batch_norm_momentum
            )
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
            x = drop(x)
            x = act(x)
            x = bn(x)
        return x
