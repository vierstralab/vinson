import os
from glob import glob

import random
import numpy as np

from argparse import ArgumentParser

import torch
from vinson.datamodules.cell_classifier import SeqEmbedDataModule

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from vinson.models.sequence import (
    CellEmbedding,
    BassetTrunkEmbed,
    EmbedModel,
)

from vinson.lr import CosineAnnealingWarmupRestarts


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

def main(args):
    """ """
    sample_genotype_file = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/output/meta+sample_ids.tsv"
    genotype_file = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/output/all_variants_stats.bed.gz"

    train_samples_files = glob(args.train_samples_files_pattern)
    train_samples_negatives_files = glob(args.train_samples_neg_files_pattern)

    valid_samples_file = args.valid_samples_file
    valid_samples_negatives_file = args.valid_samples_neg_file

    dataset_kwargs = dict(
        sample_genotype_file=sample_genotype_file,
        genotype_file=genotype_file,
        negative_samples_rate=args.negative_samples_rate,
        negative_samples_weight=args.negative_weight,
        clip_density=args.clip_density,
        min_bg=args.min_bg,
    )

    train_dataset_kwargs = dict(
        reverse_complement=True, jitter=args.jitter, noise=args.noise
    )

    valid_dataset_kwargs = dict(reverse_complement=False, jitter=0, noise=0)

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if args.accelerator == "gpu" else False,
        drop_last=True,
    )

    # DataModule to handle datasets updates and dataloader instatiation
    datamodule = SeqEmbedDataModule(
        train_samples_files,
        train_samples_negatives_files,
        valid_samples_file,
        valid_samples_negatives_file,
        args.embeddings_file,
        args.fasta_file,
        {**train_dataset_kwargs, **dataset_kwargs},
        {**valid_dataset_kwargs, **dataset_kwargs},
        dataloader_kwargs,
        worker_init_fn=set_worker_seed,
    )

    # Optimizer & LR scheduler
    optimizer = torch.optim.AdamW
    lr_scheduler = CosineAnnealingWarmupRestarts

    # TODO: Make these parameters settable via CLI
    lr_scheduler_kwargs = dict(
        max_lr=args.lr_max,
        min_lr=args.lr_min,
        warmup_steps=args.lr_warmup_steps,
        first_cycle_steps=args.lr_cycle_steps,
        cycle_mult=1,
        gamma=args.lr_decay,
        last_epoch=-1,
    )

    # Create trunk model
    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    # Create lightning module
    model = EmbedModel(
        trunk_model,
        embed_model,
        regression=args.regression,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )

    # Initialize model
    model.init_model()

    logger = CSVLogger(os.path.join(args.outdir, "logs"))

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.005, patience=10),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
            dirpath=os.path.join(
                args.outdir,
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
        accelerator=args.accelerator,
        strategy=args.strategy,
        num_nodes=args.nodes,
        devices=args.devices,
        log_every_n_steps=100,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
    )

    if args.checkpoint:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--run_id",
        default=None,
        help="Unique identifier for the training run.",
    )

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

    # ----------------------
    parser.add_argument(
        "--regression",
        action="store_true",
        default=False,
        help="Use regression mode instead of classification.",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=5,
        help="Maximum number of bases to randomly shift the region for augmentation.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise added to embeddings.",
    )
    parser.add_argument(
        "--negative_weight",
        type=float,
        default=1,
        help="Loss weight assigned to negative samples.",
    )
    parser.add_argument(
        "--negative_samples_rate",
        type=int,
        default=1,
        help="Number of negatives to sample per positive.",
    )
    parser.add_argument(
        "--clip_density",
        type=float,
        default=20,
        help="Clip densities to this value.",
    )
    parser.add_argument(
        "--min_bg",
        type=float,
        default=0.1,
        help="Minimum background level.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr_max", type=float, default=0.0005, help="Maximum learning rate."
    )
    parser.add_argument(
        "--lr_min", type=float, default=0.000005, help="Minumum learning rate."
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=5_000, help="LR warmup steps."
    )
    parser.add_argument(
        "--lr_cycle_steps", type=int, default=50_000, help="LR cosine period (steps)."
    )
    parser.add_argument("--lr_decay", type=float, default=0.9, help="LR decay rate.")
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.2,
        help="Fraction of an epoch between validation checks.",
    )

    # -------------- inputs ------------------
    parser.add_argument(
        "embeddings_file", type=str, help="Embeddings file.",
    )
    parser.add_argument(
        "fasta_file", type=str, help="FASTA file",
    )
    parser.add_argument(
        "train_samples_files_pattern",
        type=str,
        help="Glob pattern for training sample files.",
    )
    parser.add_argument(
        "train_samples_neg_files_pattern",
        type=str,
        help="Glob pattern for training negative sample files.",
    )
    parser.add_argument(
        "valid_samples_file", type=str, help="Path to validation sample file."
    )
    parser.add_argument(
        "valid_samples_neg_file",
        type=str,
        help="Path to validation negative sample file.",
    )

    args = parser.parse_args()

    try:
        os.mkdir(args.outdir)
    except OSError:
        pass

    # Set global seed
    set_global_seed(args.seed)

    # Main function
    main(args)
