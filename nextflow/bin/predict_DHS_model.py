import torch
import numpy as np
import argparse

from torch.utils.data import DataLoader
import lightning as L

from vinson.from_config import read_configs, dhs_model_from_config
from vinson.utils.data_formatting import extract_data_from_h5
from vinson.datasets.sequence import SequenceEmbedDataset

from genome_tools.data.anndata import read_zarr_backed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DHS model")
    parser.add_argument("h5_data", type=str, help="Path to DHS dataset (.h5 file)")

    parser.add_argument("anndata", type=str, help="Path to full AnnData file")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("--genotype_file", type=str, default=None, help="Path to TABIX indexed genotype file (optional)")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--output", type=str, required=True, help="Path to save model predictions (.npy file)")
    args = parser.parse_args()

    adata = read_zarr_backed(args.anndata)
    
    model_config = read_configs(args.model_config_path)

    dataset_kwargs: dict = model_config['data_params']

    dataset_kwargs.update(
        dict(
            reverse_complement=False,
            jitter=0,
            noise=0,
        )
    )

    vinson_data = extract_data_from_h5(
        h5_file=args.h5_data,
        ref_adata=adata,
    )
    
    dataset = SequenceEmbedDataset(
        data=vinson_data,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        **dataset_kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    model_predict = dhs_model_from_config(
        model_config,
        checkpoint_path=args.model_checkpoint,
    ).eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        enable_checkpointing=False,
    )

    y_hat_all = trainer.predict(model_predict, dataloaders=dataloader)
    y_hat_all = torch.cat(y_hat_all).cpu().numpy()

    np.save(args.output, y_hat_all)
