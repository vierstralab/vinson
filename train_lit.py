import sys, os

from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from vinson.dataset import SequenceEmbeddingDataset

from vinson.model import (
    CellEmbedding,
    BassetTrunkEmbed,
    EmbedModel,
)

def main(args):
    embeddings_file = "/home/jvierstra/proj/vinson/data/embeddings.tsv"
    read_depth_file = "/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/total_cutcounts.tsv"
    fasta_file = "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"

    train_dataset = SequenceEmbeddingDataset(
        args.train_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        reverse_complement=True,
        jitter=5,
        noise=0,
    )

    valid_dataset = SequenceEmbeddingDataset(
        args.val_file,
        embeddings_file,
        read_depth_file,
        fasta_file,
        reverse_complement=False,
        jitter=0,
        noise=0,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    embed = CellEmbedding(n_inputs=637, n_layers=1)
    trunk = BassetTrunkEmbed(embed)
    model = EmbedModel(trunk, embed, regression=args.regression)

    model.init_model()

    logger = CSVLogger(os.path.join(args.outdir, "logs"))

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.001, patience=50),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
            dirpath=os.path.join(
                args.outdir,
                "checkpoints",
            ),
            save_top_k=-1,
            save_last="link",
        ),
    ]

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=100,
        accelerator="gpu",
        strategy="ddp",
        num_nodes=args.nodes,
        devices=args.devices,
        log_every_n_steps=100,
        val_check_interval=0.25,
        gradient_clip_val=1.0
    )

    trainer.fit(model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--regression", action="store_true", default=False)
    parser.add_argument("train_file", help="training dataset in hdf5 format")
    parser.add_argument("val_file", help="validation dataset in hdf5 format")

    args = parser.parse_args()  

    try:
        os.mkdir(args.outdir)
    except OSError:
        pass

    main(args)
