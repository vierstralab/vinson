import sys, os
from glob import glob
from itertools import cycle

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.models.sequence import (
    CellEmbedding,
    BassetTrunkEmbed,
    EmbedModel,
)

from vinson.lr import CosineAnnealingWarmupRestarts


class SeqEmbedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_samples_files,
        train_samples_negative_files,
        val_samples_file,
        val_samples_negative_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        train_dataset_kwargs={},
        val_dataset_kwargs={},
        dataloader_kwargs={},
    ):
        super(SeqEmbedDataModule, self).__init__()

        self.embeddings_file = embeddings_file
        self.read_depth_file = read_depth_file
        self.fasta_file = fasta_file

        assert len(train_samples_files) == len(train_samples_negative_files), (
            "Train samples and negative samples files must have same length!"
        )

        self.train_samples_files = train_samples_files
        self.train_samples_negative_files = train_samples_negative_files

        self.val_samples_file = val_samples_file
        self.val_samples_negative_file = val_samples_negative_file

        self.train_dataset_kwargs = train_dataset_kwargs
        self.val_dataset_kwargs = val_dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs

        self.train = None
        self.val = None

        self.train_file_cycler = cycle(range(len(self.train_samples_files)))

    def setup(self, stage):
        self.val = SequenceEmbedDataset(
            self.val_samples_file,
            self.embeddings_file,
            self.read_depth_file,
            self.fasta_file,
            negative_samples_file=self.val_samples_negative_file,
            **self.val_dataset_kwargs,
        )

    def train_dataloader(self):
        # Cycle to next file index
        i = next(self.train_file_cycler)
        # Create new dataset
        self.train = SequenceEmbedDataset(
            self.train_samples_files[i],
            self.embeddings_file,
            self.read_depth_file,
            self.fasta_file,
            negative_samples_file=self.train_samples_negative_files[i],
            **self.train_dataset_kwargs,
        )
        # Create new dataloader
        return DataLoader(self.train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        # Reset random seed
        self.val.reset_random_state()
        # Create new dataloader
        return DataLoader(self.val, shuffle=False, **self.dataloader_kwargs)


def main(args):
    """ """
    embeddings_file = "/home/jvierstra/proj/vinson/data/embeddings.tsv"
    read_depth_file = "/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/total_cutcounts.tsv"
    fasta_file = "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
    sample_genotype_file = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v4/output/meta+sample_ids.tsv"
    genotype_file = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v4/output/all_variants_stats.bed.gz"

    train_samples_files = glob(args.train_samples_files_pattern)
    train_samples_negatives_files = glob(args.train_samples_neg_files_pattern)

    val_samples_file = args.val_samples_file
    val_samples_negatives_file = args.val_samples_neg_file

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

    val_dataset_kwargs = dict(reverse_complement=False, jitter=0, noise=0, seed=0)

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # DataModule to handle datasets updates and dataloader instatiation
    datamodule = SeqEmbedDataModule(
        train_samples_files,
        train_samples_negatives_files,
        val_samples_file,
        val_samples_negatives_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        {**train_dataset_kwargs, **dataset_kwargs},
        {**val_dataset_kwargs, **dataset_kwargs},
        dataloader_kwargs,
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
            save_top_k=3,
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
        "--nodes", type=int, default=1, help="Number of nodes for distributed training."
    )
    parser.add_argument(
        "--devices", type=int, default=4, help="Number of devices (GPUs/CPUs) per node."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for logs and checkpoints.",
    )
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
        default=0.1,
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
        default=5,
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
        "--checkpoint", type=str, help="Path to checkpoint.", default=None
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
        "val_samples_file", type=str, help="Path to validation sample file."
    )
    parser.add_argument(
        "val_samples_neg_file",
        type=str,
        help="Path to validation negative sample file.",
    )

    args = parser.parse_args()

    try:
        os.mkdir(args.outdir)
    except OSError:
        pass

    main(args)
