import os
import random
import numpy as np
import torch
import lightning as L
import anndata as ad
import sys

# from vinson.models.sequence import CellEmbedding, BassetTrunkEmbed, EmbedModel
# from vinson.models.variant import VariantEmbedModel

from vinson.models.helpers import make_legnet_model, make_dhs_model, make_variant_model

from vinson.datamodules.sequence import SeqEmbedDataModule, SeqEmbedVariantDataModule
from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.datasets.variant import VariantEmbedDataset

from vinson.utils.data_formatting import extract_data_from_h5

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger

try:
    from dnase_legnet.legnet_embed_cnn import LegNetEmbedInCNN
except ImportError:
    print("Please install dnase_legnet to use LegNet models.", file=sys.stderr)
    sys.exit(1)


def _parse_scheduler_and_optimizer(config):
    scheduler_name = config['hparams'].get("lr_scheduler")
    scheduler_kwargs = config['hparams'].get("lr_scheduler_kwargs", {})
    optimizer_kwargs = config['hparams']["optimizer_kwargs"]
    return {
        'lr_scheduler': scheduler_name,
        'lr_scheduler_kwargs': scheduler_kwargs,
        'optimizer_kwargs': optimizer_kwargs
    }

# dataset util functions
def model_from_config(config, checkpoint_path=None):
    # TODO: add loading from checkpoint
    model_type = config['model_type']

    if model_type not in {"dhs", "variant", "legnet_dhs"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_kwargs = _parse_scheduler_and_optimizer(config)
    model_kwargs = {**config["model_kwargs"], **model_kwargs} # merge model kwargs
    if model_type == "legnet_dhs":
        return make_legnet_model(config)
    elif model_type == "dhs":
        return make_dhs_model(config["model_arch"], **model_kwargs)
    elif model_type == "variant":
        return make_variant_model(config["model_arch"], **model_kwargs)


#take in config to determine model type
def dataset_from_h5(
    h5_file: str,
    ref_adata: ad.AnnData,
    fasta_file: str,
    config,
    genotype_file: str = None,
    **dataset_kwargs
):
    """
    Initialize dataset.
    Args:
        h5_file (str): Path to the H5 file.
        fasta_file (str): Path to the FASTA file.
        genotype_file (str, optional): Path to the genotype file.
        **dataset_kwargs: Additional arguments for dataset.
    """
    if config["model_type"] == 'variant':
        data, embeddings_df = extract_data_from_h5(h5_file, ref_adata=ref_adata, is_variant=True)
        dataset = VariantEmbedDataset(
            data=data,
            embeddings_df=embeddings_df,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            **dataset_kwargs,
        )
    else:
        data, embeddings_df = extract_data_from_h5(h5_file, ref_adata=ref_adata, is_variant=False)
        dataset = SequenceEmbedDataset(
            data=data,
            embeddings_df=embeddings_df,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            **dataset_kwargs,
        )

    return dataset


def datamodule_from_config(
        config,
        anndata_file,
        fasta_file,
        genotype_file,
        **dataloader_kwargs
    ):
    """
    Initialize dataloaders.
    Args:
        config (dict): Configuration dictionary. See read_configs and default config for format.
        anndata_file (str): Path to the AnnData file.
        fasta_file (str): Path to the FASTA file.
        genotype_file (str): Path to the genotype file.
        batch_size (int): Batch size for dataloaders.
        **dataloader_kwargs: Additional arguments for dataloaders.
    """
    train_dataset_kwargs = {
        **config['data_params'],
        **config['train_augmentation_kwargs'],
    }

    valid_dataset_kwargs = {
        **config['data_params'],
        **config['validation_augmentation_kwargs'],
    }
    
    dataloader_kwargs = {
        'batch_size': config['hparams']['batch_size'],
        **dataloader_kwargs,
    }

    # DataModule to handle datasets updates and dataloader init
    if config.get("model_type") == "variant":
        return SeqEmbedVariantDataModule(
            anndata_file=anndata_file,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            train_dataset_kwargs=train_dataset_kwargs,
            valid_dataset_kwargs=valid_dataset_kwargs,
            **dataloader_kwargs,
            )
    else:
        return SeqEmbedDataModule(
            anndata_file=anndata_file,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            train_dataset_kwargs=train_dataset_kwargs,
            valid_dataset_kwargs=valid_dataset_kwargs,
            **dataloader_kwargs,
        )

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
        **trainer_kwargs,
    )
    return trainer


def fit_model(model, trainer: L.Trainer, datamodule, checkpoint=None):
    """Fit a model with optional checkpoint resume."""
    if checkpoint is not None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        trainer.fit(model, datamodule=datamodule)