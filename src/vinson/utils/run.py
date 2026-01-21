import os
import random
import numpy as np
import torch
import lightning as L
import anndata as ad
import sys

from vinson.models.sequence import CellEmbedding, BassetTrunkEmbed, EmbedModel, VariantEmbedModel
from vinson.datamodules.sequence import SeqEmbedDataModule, SeqEmbedVariantDataModule
from vinson.datasets.sequence import SequenceEmbedDataset, VariantEmbedDataset
from vinson.utils.data_formatting import extract_data_from_h5

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import CSVLogger


# dataset util functions
def model_from_config(config, checkpoint_path=None):
    # TODO: add model configuration to config
    model_type = config['model_type']
    hparams = config["hparams"]
    scheduler_name = hparams.get("lr_scheduler")
    scheduler_kwargs = hparams.get("lr_scheduler_kwargs", {})
    optimizer_kwargs = hparams.get("optimizer_kwargs", {"lr": hparams.get("lr")})
    trunk_weights = config.get("trunk_weights", None)

    # legacy fix
    if model_type == "regression":
        model_type = "dhs"

    if model_type not in {"dhs", "variant", "legnet_dhs"}:
        raise ValueError(f"Unsupported model type: {model_type}")
    if model_type == "legnet_dhs":
        try:
            from dnase_legnet.legnet_embed_cnn import LegNetEmbedInCNN
        except ImportError:
            print("Please install dnase_legnet to use LegNet models.", file=sys.stderr)
            sys.exit(1)

        if checkpoint_path:
            return LegNetEmbedInCNN.load_from_checkpoint(
                checkpoint_path,
                inference_mode=False
            )

        return LegNetEmbedInCNN(
            model_kws=config["model_arch"],
            hparams=config['hparams'],
            # hparams={
            #     "lr_scheduler": scheduler_name,
            #     "lr_scheduler_kwargs": scheduler_kwargs,
            #     "optimizer_kwargs": optimizer_kwargs,
            # },
            **config["model_kwargs"],
        )

    # --- STANDARD EMBED/TRUNK PATH ---
    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    # --- CHECKPOINT LOADING ---
    if checkpoint_path:
        load_cls = VariantEmbedModel if model_type == "variant" else EmbedModel
        return load_cls.load_from_checkpoint(
            checkpoint_path,
            trunk=trunk_model,
            embed=embed_model,
        )

    # --- MODEL INITIALIZATION (NO CHECKPOINT) ---
    if model_type == "variant":
        model = VariantEmbedModel(
            trunk=trunk_model,
            embed=embed_model,
            optimizer_kwargs=optimizer_kwargs,   # contains lr
        )
    else:
        model = EmbedModel(
            trunk=trunk_model,
            embed=embed_model,
            regression=True,
            lr_scheduler=scheduler_name,
            lr_scheduler_kwargs=scheduler_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            **config["model_kwargs"],
        )

    model.init_model()

    # ---- LOAD PRETRAINED TRUNK WEIGHTS ----
    if trunk_weights is not None:
        print(f"[INFO] Loading pretrained trunk weights from {trunk_weights}")

        ckpt = torch.load(trunk_weights, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        # Only load trunk + embed weights
        trunk_state = {k: v for k, v in state.items() if k.startswith("trunk.")}
        embed_state = {k: v for k, v in state.items() if k.startswith("embed.")}
        model.load_state_dict({**trunk_state, **embed_state}, strict=False)


    return model


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
        data = extract_data_from_h5(h5_file, ref_adata=ref_adata, is_variant=True)
        dataset = VariantEmbedDataset(
            data=data,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            **dataset_kwargs,
        )
    else:
        data = extract_data_from_h5(h5_file, ref_adata=ref_adata, is_variant=False)
        dataset = SequenceEmbedDataset(
            data=data,
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
        **config.get('data_params', {}),  # optional
        **config.get('train_augmentation_kwargs', {}),
    }

    valid_dataset_kwargs = {
        **config.get('data_params', {}),  # optional
        **config.get('validation_augmentation_kwargs', {}),
    }
    
    dataloader_kwargs = {
        'batch_size': config.get('hparams', {}).get('batch_size', 64),
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
            **dataloader_kwargs
        )
    else:
        return SeqEmbedDataModule(
            anndata_file=anndata_file,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            train_dataset_kwargs=train_dataset_kwargs,
            valid_dataset_kwargs=valid_dataset_kwargs,
            **dataloader_kwargs
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
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.005,
            patience=10,
            check_on_train_epoch_end=False,
        ),
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

    # callbacks = [
    #     EarlyStopping(monitor="val_loss", mode="min", min_delta=0.005, patience=10),
    #     ModelCheckpoint(
    #         monitor="val_loss",
    #         mode="min",
    #         filename="{epoch}-{step}-{val_loss:.2f}",
    #         dirpath=os.path.join(outdir, "checkpoints"),
    #         save_top_k=5,
    #         save_last=True,
    #     ),
    #     LearningRateMonitor(),
    # ]

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=10,
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