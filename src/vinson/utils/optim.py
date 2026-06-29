from typing import Dict, Any, Optional, Union
import torch

from .hparams import get_lr_scheduler_cls


# TODO: move parsing to from_config.py
def configure_optimizer(
    *,
    module_parameters,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_cls=torch.optim.AdamW,
    lr_scheduler: Optional[str] = None,
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
    """
    Configure optimizer and optional LR scheduler for Lightning.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = optimizer_cls(
        module_parameters,
        **optimizer_kwargs,
    )

    if lr_scheduler is None or str(lr_scheduler).lower() == "none":
        return {
            "optimizer": optimizer
        }


    if lr_scheduler_kwargs is None:
        lr_scheduler_kwargs = {}

    scheduler_cls = get_lr_scheduler_cls(lr_scheduler)
    scheduler = scheduler_cls(optimizer, **lr_scheduler_kwargs)

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "lr",
        },
    }
