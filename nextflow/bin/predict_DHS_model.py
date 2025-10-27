import torch
import sys
import numpy as np
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader
from vinson.utils.run import model_from_config as load_vinson_model
from vinson.utils.run import read_configs
from vinson.utils.run import dataset_from_h5_and_config

from genome_tools.data.anndata import read_zarr_backed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_legnet_model(checkpoint_path, device):
    try:
        from dnase_legnet.legnet_embed_cnn import LegNetEmbedinCNN
    except ImportError:
        print("Please install dnase_legnet to use LegNet models.", file=sys.stderr)
        sys.exit(1)

    model = LegNetEmbedinCNN.load_from_checkpoint(checkpoint_path, map_location=device).eval()
    return model


@torch.inference_mode()
def load_and_predict(batch, model):    
    X_seq      = batch["ohe_seq"].to(device, non_blocking=True)
    X_embed    = batch["embed"].to(device, non_blocking=True)
    y_ = model(X_seq, X_embed).squeeze().detach().cpu().numpy()
    return y_


def main():
    parser = argparse.ArgumentParser(description="Predict DHS model")
    parser.add_argument("h5_data", type=str, help="Path to DHS dataset (.h5 file)")
    parser.add_argument("anndata", type=str, help="Path to full AnnData file")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("--genotype_file", type=str, default=None, help="Path to TABIX indexed genotype file (optional)")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument(
        "--model_type", type=str, default="vinson", choices=["vinson", "legnet"],
        help="Type of model to use for prediction"
    )
    parser.add_argument("--output", type=str, required=True, help="Path to save model predictions (.npy file)")
    args = parser.parse_args()

    adata = read_zarr_backed(args.anndata)

    model_config = read_configs(args.model_config_path)

    dataset_kwargs = dict(
        reverse_complement=False,
        jitter=0,
        noise=0,
    )

    dataset = dataset_from_h5_and_config(
        config=model_config,
        h5_file=args.h5_data,
        embeddings_df=adata.obsm['motif_embeddings'],
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

    if args.model_type == "vinson":
        model_predict = load_vinson_model(model_config, args.checkpoint)
    elif args.model_type == "legnet":
        model_predict = load_legnet_model(args.checkpoint, device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model_predict.to(device).eval()

    y_hat_all = np.concatenate(
        [load_and_predict(batch, model_predict) for batch in tqdm(dataloader)]
    )
    np.save(args.output, y_hat_all)

if __name__ == "__main__":
    main()