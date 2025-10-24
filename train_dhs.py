import os
import sys
import random
import numpy as np
import yaml
from argparse import ArgumentParser
from datetime import datetime

import torch
import anndata as ad
import mergedeep
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from vinson.datamodules.sequence import SeqEmbedDataModule
from vinson.models.sequence import (
    CellEmbedding,
    BassetTrunkEmbed,
    EmbedModel,
)

from vinson.lr import CosineAnnealingWarmupRestarts


from vinson.utils import generate_run_name, read_yaml_config


def set_global_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)

    # Numpy's random module
    np.random.seed(seed)

    # PyTorch seed for CPU
    torch.manual_seed(seed)

    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)

    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id):
    # Set seed for Python and NumPy in each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# class IterateDataModule(L.Callback):
#     def on_train_epoch_end(self, trainer, pl_module):
#         trainer.datamodule.iterate_train_dataset()


def init_trainer(
        outdir,
        accelerator,
        strategy,
        nodes,
        devices,
        logger_type,
        val_check_interval,
    ):
    assert logger_type in ["csv"], "Only 'csv' logger is currently supported."
    logger = CSVLogger(os.path.join(outdir, "logs"))

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.005, patience=10),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
            dirpath=os.path.join(
                outdir,
                "checkpoints",
            ),
            save_top_k=5,
            save_last="link",
        ),
        LearningRateMonitor(),
    ]

    # Lightning trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=100,
        accelerator=accelerator,
        strategy=strategy,
        num_nodes=nodes,
        devices=devices,
        val_check_interval=val_check_interval,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
    )
    return trainer


def main(
        config,
        anndata_file,
        fasta_file,
        genotype_file,
        trainer: L.Trainer,
        num_workers=None,
        accelerator="gpu",
        checkpoint=None,
    ):

    train_dataset_kwargs = {
        **config['data_params'],
        **config['train_augmentation_kwargs'],
    }

    valid_dataset_kwargs = {
        **config['data_params'],
        **config['validation_augmentation_kwargs'],
    }

    batch_size = config['hparams']['batch_size']

    dataloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if accelerator == "gpu" else False,
        drop_last=True,
    )
    
    adata = ad.read_h5ad(anndata_file)
    # DataModule to handle datasets updates and dataloader instatiation
    datamodule = SeqEmbedDataModule(
        adata=adata,
        fasta_file=fasta_file,
        genotype_file=genotype_file,
        train_dataset_kwargs=train_dataset_kwargs,
        valid_dataset_kwargs=valid_dataset_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        worker_init_fn=set_worker_seed,
    )

    # Optimizer & LR scheduler
    optimizer = torch.optim.AdamW
    lr_scheduler = CosineAnnealingWarmupRestarts

    lr_scheduler_kwargs = config["hparams"]["lr_scheduler_kwargs"]

    # Create trunk model, maybe move to config later
    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    model = EmbedModel(
        trunk=trunk_model,
        embed=embed_model,
        regression=config["model_type"] == "regression",
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )

    # Initialize model
    model.init_model()

    if checkpoint is not None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "anndata_file",
        type=str,
        help="Input AnnData file.",
    )

    parser.add_argument(
        "fasta_file", type=str, help="FASTA file",
    )

    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Unique identifier for the training run. Generated if not provided.",
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None, 
        help='Path to YAML config file (see default config for format). If provided, overrides default parameters.'
    )

    parser.add_argument(
        "--genotype_file",
        type=str,
        default=None,
        help="Path to Tabix indexed genotype file.",
    )

    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint.", default=None,
    )

    parser.add_argument(
        "--seed", type=int, help="Random seed", default=42,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for logs and checkpoints.",
    )
    # --- torch multi-gpu setup ---

    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes for distributed training."
    )
    parser.add_argument(
        "--devices", type=int, default=4, help="Number of devices (GPUs/CPUs) per node."
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Type of accelerator to use (e.g., 'gpu', 'cpu').",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Distributed training strategy (e.g., 'ddp', 'auto').",
    )

    args = parser.parse_args()

    # Setup output
    run_name = args.run_name.strip() or "vinson" #or generate_run_name() generates unique name in each subprocess. currently done outside of script

    outdir = os.path.join(args.outdir, run_name)

    os.makedirs(outdir, exist_ok=True)
    
    # Config processing
    # TODO: move to utils
    default_config_path = os.path.dirname(os.path.abspath(__file__)) + "/default_train_dhs.config.yaml"
    config = read_yaml_config(default_config_path)
    if args.config is not None:
        update_config = read_yaml_config(args.config)
        mergedeep.merge(config, update_config, strategy=mergedeep.Strategy.REPLACE)

    config['command'] = " ".join(["python"] + sys.argv)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(outdir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    # Set global seed
    set_global_seed(args.seed)

    # Initialize trainer
    trainer = init_trainer(
        outdir,
        accelerator=args.accelerator,
        strategy=args.strategy,
        nodes=args.nodes,
        devices=args.devices,
        logger_type=config["logging_params"]["logger_type"],
        val_check_interval=config["logging_params"]["val_check_interval"],
    )

    # Main function
    main(
        config, 
        anndata_file=args.anndata_file,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        trainer=trainer,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        checkpoint=args.checkpoint,
    )
