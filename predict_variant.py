import torch
import numpy as np
import argparse
import anndata as ad

from torch.utils.data import DataLoader
import lightning as L

from vinson.from_config import read_configs, variant_model_from_config
from vinson.utils.data_formatting import extract_variant_data_from_anndata
from vinson.datasets.variant import VariantEmbedDataset

from genome_tools.data.anndata import read_zarr_backed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DHS model")
    parser.add_argument("anndata", type=str, help="Path to full AnnData file")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("--genotype_file", type=str, default=None, help="Path to TABIX indexed genotype file (optional)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=str, required=True, help="Path to save model predictions (.npy file)")
    args = parser.parse_args()
    
    model_config = read_configs(args.model_config_path)

    dataset_kwargs: dict = model_config.get('data_params', {}).copy()

    dataset_kwargs.update({
        "reverse_complement": False,
        "jitter": 0,
        "noise": 0,
    })

    adata = ad.read_h5ad(args.anndata)

    vinson_data = extract_variant_data_from_anndata(
        train_adata=adata,
        suffix='epoch_1',
    )
    
    dataset = VariantEmbedDataset(
        data=vinson_data,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        flip_alleles=False,
        **dataset_kwargs,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        drop_last=False,
    )


    model_predict = variant_model_from_config(
        model_config,
        checkpoint_path=args.model_checkpoint,
    ).eval()

    trainer = L.Trainer(
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    y_hat_all = trainer.predict(model_predict, dataloaders=dataloader)
    y_hat_all = torch.cat(y_hat_all).cpu().numpy()

    np.save(args.output, y_hat_all)
    

    
