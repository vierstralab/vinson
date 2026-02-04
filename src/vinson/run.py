import os
import random
import numpy as np
import torch
import lightning as L

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger


#functions for training
def set_global_seed(seed: int = 42):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_worker_seed(worker_id: int):
    """Set seed for each dataloader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_single_gpu_trainer(
    
):
    # TODO: implement single gpu trainer
    pass


def init_multigpu_trainer(
    outdir,
    accelerator="gpu",
    strategy="ddp",
    nodes=1,
    devices=1,
    logger_type="csv",
    val_check_interval=1.0,
    **trainer_kwargs,
):
    """Initialize a Lightning Trainer for multi-GPU runs."""
    assert logger_type in ["csv"], "Only 'csv' logger is currently supported."

    logger = CSVLogger(os.path.join(outdir, "logs"))

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.005, patience=10),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
            dirpath=os.path.join(outdir, "checkpoints"),
            save_top_k=5,
            save_last=True,
        ),
        LearningRateMonitor(),
    ]

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=20,
        accelerator=accelerator,
        strategy=strategy,
        num_nodes=nodes,
        devices=devices,
        val_check_interval=val_check_interval,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
        **trainer_kwargs,
    )
    return trainer


def fit_model(model, trainer: L.Trainer, datamodule, checkpoint=None):
    """Fit a model with optional checkpoint resume."""
    if checkpoint is not None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        trainer.fit(model, datamodule=datamodule)
