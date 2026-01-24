import torch

ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "selu": torch.nn.SELU,
}