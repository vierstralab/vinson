import sys, os

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

from vinson.dataset import VariantEmbedDataset

from vinson.model import (
    CellEmbedding,
    BassetTrunkEmbed,
    VariantEmbedModel,
)


def main(args):
    """ """
    embeddings_file = "/home/jvierstra/proj/vinson/data/embeddings.tsv"
    fasta_file = "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"

    train_dataset = VariantEmbedDataset(
        args.train_file,
        embeddings_file,
        fasta_file,
        reverse_complement=True,
        jitter=args.jitter,
        noise=args.noise,
    )

    valid_dataset = VariantEmbedDataset(
        args.val_file,
        embeddings_file,
        fasta_file,
        reverse_complement=False,
        jitter=0,
        noise=0,
    )

    dataloader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        **dataloader_kwargs,
    )

    # Create model
    embed = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk = BassetTrunkEmbed(embed.n_outputs)
    model = VariantEmbedModel(trunk, embed)

    # Initialize model
    model.init_model()

    # Load model trunk weights
    if args.trunk_weights:
        print(f"Loading weights from pre-trained model: {args.trunk_weights}")
        pretrained_state_dict = torch.load(
            args.trunk_weights,
            map_location=torch.device("cpu"),
        )["state_dict"]

        embed_pretrained_dict = {
            k: v for k, v in pretrained_state_dict.items() if "embedding." in k
        }
        trunk_pretrained_dict = {
            k: v for k, v in pretrained_state_dict.items() if "trunk." in k
        }

        model_dict = model.state_dict()
        model_dict.update({**embed_pretrained_dict, **trunk_pretrained_dict})

        model.load_state_dict(model_dict, strict=False)

    # Configure trainer logger & callbacks
    logger = CSVLogger(os.path.join(args.outdir, "logs"))

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0005, patience=50),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.4f}",
            dirpath=os.path.join(
                args.outdir,
                "checkpoints",
            ),
            save_top_k=3,
            save_last="link",
        ),
        LearningRateMonitor(),
    ]

    # Trainer
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
    )

    # Run trainer
    trainer.fit(model, train_dataloader, valid_dataloader)


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
        "--jitter",
        type=int,
        default=25,
        help="Maximum number of bases to randomly shift the region for augmentation.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Standard deviation of Gaussian noise added to embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0005, help="Learning rate (not implemented yet)."
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.5,
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
        "--trunk_weights",
        type=str,
        default=None,
        help="Checkpoint of pre-trained model to initialize trunk weights.",
    )
    parser.add_argument("train_file", help="Training dataset in hdf5 format")
    parser.add_argument("val_file", help="Validation dataset in hdf5 format")

    args = parser.parse_args()

    try:
        os.mkdir(args.outdir)
    except OSError:
        pass

    main(args)
