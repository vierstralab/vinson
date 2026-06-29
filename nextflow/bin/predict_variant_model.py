import torch
import numpy as np
import argparse
import anndata as ad
from torch.utils.data import DataLoader
import lightning as L
from tqdm import tqdm

from vinson.from_config import read_configs, variant_model_from_config
from vinson.utils.data_formatting import extract_variant_data_from_anndata, extract_data_from_h5
from vinson.datasets.variant import VariantEmbedDataset

import h5py
import numpy as np
import pandas as pd
from vinson.utils.data_formatting.readers import sanitize_data
from vinson.utils.data_formatting.container import VinsonData

def extract_variant_data_from_h5(h5_file: str, ref_adata=None, is_variant=True) -> VinsonData:
    """
    Extract variant data from an H5 file (all datasets, no sparse groups)
    and create a VinsonData object similar to extract_variant_data_from_anndata.
    """
    with h5py.File(h5_file, 'r') as f:
        data_raw = {key: f[key][()] for key in f.keys()}

    # Convert bytes/string arrays to str
    for key, val in data_raw.items():
        if val.dtype.kind in {'S', 'O'}:
            data_raw[key] = val.astype(str)

    # Encode categorical/string columns
    encodings = {}
    for col in ['chrom', 'ref', 'alt']:
        if col in data_raw:
            codes, uniques = pd.factorize(data_raw[col])
            data_raw[col] = codes
            encodings[col] = uniques

    # Assemble final data dictionary
    data = {
        'chrom': data_raw['chrom'],
        'pos': data_raw['pos'],
        'ref': data_raw['ref'],
        'alt': data_raw['alt'],
        'ref_counts': data_raw['ref_counts'],
        'total_counts': data_raw['total_counts'],
        'BAD': data_raw['BAD'],
        'logit_es': data_raw['logit_es'],
        'sample_id':data_raw['sample_id'],
    }

    # Sanitize & create VinsonData
    data, encodings = sanitize_data(data, encodings, is_variant=is_variant)
    # embeddings_df = ref_adata.obsm['motif_embeddings'] if ref_adata is not None else None
    if ref_adata is not None:
        embeddings_df = pd.DataFrame(
            ref_adata.obsm["motif_embeddings"],
            index=ref_adata.obs_names
        )
    else:
        embeddings_df = None

    return VinsonData(
        data,
        encodings=encodings,
        embeddings_df=embeddings_df,
        is_variant=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict variants with Vinson model")
    parser.add_argument("anndata", type=str, help="Path to full AnnData file")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("--genotype_file", type=str, default=None, help="Path to TABIX indexed genotype file (optional)")
    parser.add_argument("--h5_file", type=str, default=None, help="H5 file with specific variants to predict (optional)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=str, required=True, help="Path to save model predictions (.npy file)")
    args = parser.parse_args()

    # -------------------------
    # Load config
    # -------------------------
    print("Loading config...")
    model_config = read_configs(args.model_config_path)
    dataset_kwargs = model_config.get('data_params', {}).copy()
    dataset_kwargs.update({
        "reverse_complement": False,
        "jitter": 0,
    })

    # -------------------------
    # Load data
    # -------------------------
    if args.h5_file is not None:
        print(f"Loading VinsonData from H5: {args.h5_file}")
        adata = ad.read_h5ad(args.anndata)  # need ref for embeddings
        vinson_data = extract_variant_data_from_h5(args.h5_file, ref_adata=adata, is_variant=True)
    else:
        print(f"Loading full AnnData: {args.anndata}")
        ##remove later just validation data
        adata = ad.read_h5ad(args.anndata)
        dhs_split = "val"
        adata_val = adata[
            :,
            adata.varm["split_data"] == dhs_split,
        ]
        vinson_data = extract_variant_data_from_anndata(
            train_adata=adata_val,
            suffix='epoch_1',  # adjust if needed
        )

    print(f"Extracted VinsonData length: {len(vinson_data)}")
    print(f"VinsonData keys: {list(vinson_data.keys())}")

    # -------------------------
    # Dataset & Dataloader
    # -------------------------
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

    # -------------------------
    # Load model
    # -------------------------
    print("Loading model...")
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
    print(f"Predictions shape: {y_hat_all.shape}")

    # -------------------------
    # Save output
    # -------------------------
    np.save(args.output, y_hat_all)
    print(f"Saved predictions to: {args.output}")