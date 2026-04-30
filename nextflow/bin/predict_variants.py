from vinson.from_config import read_configs, dhs_model_from_config
from vinson.postprocessing.interpretation import ModelWrapper
import torch

from vinson.utils.data_formatting import VinsonData
from torch.utils.data import DataLoader

from tqdm import tqdm
from vinson.datasets.variant import VariantInferenceDataset
from genome_tools.data.anndata import read_zarr_backed
import sys
import pandas as pd


def make_loader(variants_df, anndata, fasta_file):
    loaded_data = VinsonData(
        data=variants_df.to_dict(orient='list'),
        embeddings_df=anndata.obsm['motif_embeddings'],
        encodings={}
    )
    dataset_qtl = VariantInferenceDataset(
        loaded_data,
        fasta_file=fasta_file,
        strict_ref_check=False
    )
    return DataLoader(dataset_qtl, batch_size=128, num_workers=10)


def predict(model, dataloader):
    pred_ref_all = []
    pred_alt_all = []

    for batch in tqdm(dataloader):
        ohe_ref = batch["ohe_seq_ref"]
        ohe_alt = batch["ohe_seq_alt"]
        embed = batch["embed"]

        pred_ref = model(ohe_ref, embed)   # (B,)
        pred_alt = model(ohe_alt, embed)

        pred_ref_all.append(pred_ref.detach().cpu())
        pred_alt_all.append(pred_alt.detach().cpu())

    pred_ref_all = torch.cat(pred_ref_all).numpy()
    pred_alt_all = torch.cat(pred_alt_all).numpy()

    return pred_ref_all, pred_alt_all


if __name__ == "__main__":
    variants = pd.read_table(sys.argv[1])
    anndata = read_zarr_backed(sys.argv[2])
    fasta = sys.argv[3]

    checkpoint = sys.argv[4]
    config_path = sys.argv[5]

    name = sys.argv[6]

    dl = make_loader(
        variants_df=variants,
        anndata=anndata,
        fasta_file=fasta
    )
    model_config = read_configs(config_path)

    model_predict = dhs_model_from_config(model_config, checkpoint).eval()
    pred_ref, pred_alt = predict(model_predict, dl)
    variants["pred_ref"] = pred_ref
    variants["pred_alt"] = pred_alt

    variants.to_csv(name, sep='\t', index=False)





