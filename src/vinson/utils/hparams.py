from typing import Any, Callable, Dict, List, Union
from pandas.api.types import is_list_like
import torch

from vinson.optim.lr import CosineAnnealingWarmupRestarts


_ACTIVATIONS = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "selu": torch.nn.SELU,
}


_LR_SCHEDULERS = {
    "CosineAnnealingWarmupRestarts": CosineAnnealingWarmupRestarts,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}


def get_hidden_dims(values: Union[int, List[int]], n_layers: int | None = None):
    if is_list_like(values):
        if n_layers is not None and len(values) != n_layers:
            raise ValueError(
                f"Expected {n_layers} hidden dims, got {len(values)}"
            )
        return list(values)

    if n_layers is None:
        return [values]

    return [values] * n_layers


def get_lr_scheduler_cls(value: Union[str, Any]):
    """
    Resolve a learning-rate scheduler specification to a scheduler class.

    Parameters
    ----------
    value
        Either a string key registered in `_LR_SCHEDULERS` or a scheduler
        class itself.

    Returns
    -------
    type
        Learning-rate scheduler class. The scheduler is not instantiated.
    """
    if isinstance(value, str):
        return _sanitize_value(
            value,
            registry=_LR_SCHEDULERS,
        )
    return value


def get_activations(values: Union[List[str], str], n_expected: int):
    """
    Resolve activation function specifications to list of instantiated activation modules.

    Parameters
    ----------
    values
        Either a single activation name or a list of activation names.
        String values must be registered in `_ACTIVATIONS`.
    n_expected
        Number of activations expected. Scalars are broadcasted to this length.

    Returns
    -------
    list of torch.nn.Module
        List of instantiated activation modules.
    """
    vals = _sanitize_list(
        values,
        n_expected,
        registry=_ACTIVATIONS,
    )
    return [v() for v in vals]





def _sanitize_value(
    v: Any,
    *,
    registry: Dict[str, Any] | None = None,
):
    if registry is not None and isinstance(v, str):
        key = v
        if key in registry:
            v = registry[key]
        else:
            raise KeyError(f"{key} not in {list(registry)}")

    return v


def _sanitize_list(
    values: Union[List[Any], Any],
    n_expected: int,
    *,
    registry: Dict[str, Any] | None = None,
):
    if is_list_like(values):
        if len(values) != n_expected:
            raise ValueError(f"Expected length {n_expected}, got {len(values)}")
        values = list(values)
    else:
        values = [values] * n_expected

    return [
        _sanitize_value(v, registry=registry)
        for v in values
    ]

