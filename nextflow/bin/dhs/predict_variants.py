from vinson.from_config import read_configs, dhs_model_from_config
import torch

from vinson.utils.data_formatting import VariantVinsonData
from vinson.postprocessing.inference import score_variants_with_dhs_model
from torch.utils.data import DataLoader

from tqdm import tqdm
from vinson.datasets.variant import InferenceVariantEmbedDataset
from genome_tools.data.anndata import read_zarr_backed
import argparse
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# FIXME
def make_loaders(loaded_data, fasta_file, num_workers=10, batch_size=256):

    dataset = InferenceVariantEmbedDataset(
        loaded_data,
        fasta_file=fasta_file,
        strict_ref_check=False,
    )
    print('Datasets created.', flush=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def predict(model, dataloader):
    pred_ref_all = []
    pred_alt_all = []

    for batch in tqdm(dataloader):
        ohe_ref = batch["ohe_seq_ref"].to(device, non_blocking=True)
        ohe_alt = batch["ohe_seq_alt"].to(device, non_blocking=True)
        embed = batch["embed"].to(device, non_blocking=True)

        pred_ref = model(ohe_ref, embed)   # (B,)
        pred_alt = model(ohe_alt, embed)

        pred_ref_all.append(pred_ref.detach().cpu())
        pred_alt_all.append(pred_alt.detach().cpu())

    pred_ref_all = torch.cat(pred_ref_all).numpy()
    pred_alt_all = torch.cat(pred_alt_all).numpy()

    return pred_ref_all, pred_alt_all

def int_list(s):
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        return list(range(*parts))  # start:stop[:step]
    return [int(x) for x in s.split(",")]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict variants with DHS model")
    parser.add_argument("variant_dataset", type=str, help="Path to DHS dataset (.tsv file)")

    parser.add_argument("anndata", type=str, help="Path to full AnnData file")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("sample_id", type=str, help="Comma separated sample id(s). If multiple ids provided - averages the embedding")

    parser.add_argument("--offsets", type=int_list, default=[0],
                        help="Comma-separated offsets, e.g. -40,-20,0,20,40")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--output", type=str, required=True, help="Path to save model predictions (.npy file)")
    args = parser.parse_args()

    variants = pd.read_table(args.variant_dataset)
    anndata = read_zarr_backed(args.anndata)
    fasta = args.fasta_file

    checkpoint = args.model_checkpoint
    config_path = args.model_config_path

    name = args.output

    offsets = args.offsets
    variants['sample_id'] = args.sample_id

    sample_ids = args.sample_id.split(',')
    motif_embeddings_df = anndata.obsm['motif_embeddings'].loc[sample_ids]
    if len(sample_ids) > 1:
        motif_embeddings_df = motif_embeddings_df.mean(axis=0).rename(args.sample_id).to_frame().T

    loaded_data = VariantVinsonData.inference_from_variant(
        data=variants,
        embeddings_df=motif_embeddings_df,
        sample_ids=sample_ids,
        shifts=offsets,
        is_inference=True
    )
    dl = make_loaders(
        data=loaded_data,
        fasta_file=fasta,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    model_config = read_configs(config_path)

    model_predict = dhs_model_from_config(model_config, checkpoint).to(device).eval()

    expanded_preds = loaded_data.to_df()
    pred_ref, pred_alt = score_variants_with_dhs_model(dl, model_predict)
    expanded_preds['pred_ref'] = pred_ref
    expanded_preds['pred_alt'] = pred_alt

    expanded_preds.to_csv(name, sep='\t', index=False)
