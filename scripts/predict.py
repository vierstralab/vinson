import torch
from torch.utils.data import DataLoader
import numpy as np
from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor, TabixExtractor
import pandas as pd
import argparse
from tqdm import tqdm

from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.loss import poisson_loss, mse_loss
from vinson.models.sequence import BassetTrunkEmbed, CellEmbedding, EmbedModel
from vinson.interpretation import dinucleotide_shuffle, force_strict_ohe
from vinson.io import SamplesDensityExtractor
from vinson.vinson.utils.sequence_utils import intervals_to_ohe

"""
This script runs genome-wide predictions using a pretrained Vinson sequence +
cell embedding model. It processes evaluation samples, applies the model, and
outputs per-interval predicted and observed counts/densities. This is for the H5 validation, not for a specific cell type (use predict_cell.py for prediciton for each peak in cell type)

Workflow:
1. Loads evaluation samples (HDF5), embeddings, read depth file, reference FASTA,
   and a pretrained model checkpoint.
2. Optionally incorporates:
   - Negative control regions (BED.GZ)
   - Personalized genotype data (sample-to-genotype mapping + genotype stats BED.GZ)
3. Builds a SequenceEmbedDataset and DataLoader to efficiently batch data.
4. Runs the model in evaluation mode (no gradient computation) to predict counts
   for each interval using sequence + embedding inputs.
5. Normalizes predictions by sample read depth and adds background adjustment.
6. Collects outputs including:
   - Chromosome and midpoint of each interval
   - Sample/embedding ID
   - Predicted vs observed counts
   - Predicted vs observed total density (normalized)
7. Combines results into a dataframe and writes a TSV output file.

Outputs:
- `<output>.tsv`: per-interval table with chrom, mid, embedding_id, predicted_counts,
  target_counts, predicted_total_density, and observed_total_density.
"""

#inputs
parser = argparse.ArgumentParser(description="Run prediction with Vinson model.")
parser.add_argument("--eval-samples", required=True, help="Path to evaluation samples HDF5 file")
parser.add_argument("--embeddings", required=True, help="Path to embeddings TSV file")
parser.add_argument("--read-depths", required=True, help="Path to total cut counts file")
parser.add_argument("--fasta", required=True, help="Path to FASTA file")
parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
parser.add_argument("--output", required=True, help="Path to output TSV file")
parser.add_argument("--negatives", default=None, help="Path to BED.GZ file containing negative regions")

# Optional: personalized genome files
parser.add_argument(
    "--sample-genotype-file",
    default=None,
    help="Path to TSV file mapping samples to genotype information"
)
parser.add_argument(
    "--genotype-file",
    default=None,
    help="Path to BED.GZ file containing genotype variant stats"
)

args = parser.parse_args()


# --- Required inputs ---
samples_file = args.eval_samples
embeddings_file = args.embeddings
read_depth_file = args.read_depths
fasta_file = args.fasta
model_ckpt = args.checkpoint
output_file = args.output 
    
dataset_kwargs = dict(
    reverse_complement=False,
    jitter=0,
    noise=0,
    seed=0,
)

# Optional: personalized genome data
if args.sample_genotype_file is not None and args.genotype_file is not None:
    dataset_kwargs["sample_genotype_file"] = args.sample_genotype_file
    dataset_kwargs["genotype_file"] = args.genotype_file

# Optional: negative samples
if args.negatives is not None:
    dataset_kwargs["negative_samples_file"] = args.negatives
    dataset_kwargs["negative_samples_rate"] = 2

dataset = SequenceEmbedDataset(
    samples_file,
    embeddings_file,
    read_depth_file,
    fasta_file,
    **dataset_kwargs,
)

dataloader = DataLoader(
    dataset,
    batch_size=256,   
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
)

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
embed = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
trunk = BassetTrunkEmbed(embed.n_outputs)
model = EmbedModel(trunk, embed, regression=True)

#load model
pretrained_state_dict = torch.load(
    model_ckpt,
    map_location=torch.device(device),
)["state_dict"]

model.load_state_dict(pretrained_state_dict)

model = model.to(device)
model = model.eval()

# predict with metadata
all_preds = []
all_targets = []
all_chr = []
all_mid = []
all_embed_id = []
all_pred_density = []
all_actual_density = []

#store sample id
#store some sort of chr marker for dhs
#no grad means dont have to detach everytime
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
        X_seq = batch["ohe_seq"].to(device, non_blocking=True)
        X_embed = batch["embed"].to(device, non_blocking=True)
        density = batch["density"].to(device, non_blocking=True)
        read_depth = batch["read_depth"].to(device, non_blocking=True)
        bg = batch['bg'].to(device, non_blocking=True)
        chrom = batch["chrom"] 
        mid = batch["mid"]      
        embed_id = batch["sample_id"] 
        
        y_pred = model(X_seq, X_embed).squeeze()
        pred_counts = (torch.exp(y_pred) / 1e6 * read_depth) + bg
        target_counts = density / 1e6 * read_depth
        bg_density = bg / read_depth * 1e6
        pred_total_density = torch.exp(y_pred) + bg_density
        
        all_preds.append(pred_counts.cpu())
        all_targets.append(target_counts.cpu())
        all_pred_density.append(pred_total_density.cpu())
        all_actual_density.append(density.cpu())
        all_chr.extend(chrom) 
        all_mid.extend(mid)
        all_embed_id.extend(embed_id)

# === Combine all into DataFrame ===
all_preds = torch.cat(all_preds).cpu().numpy()
all_targets = torch.cat(all_targets).cpu().numpy()
all_pred_density = torch.cat(all_pred_density).cpu().numpy()
all_act_density = torch.cat(all_actual_density).cpu().numpy()


output_df = pd.DataFrame({
    "chrom": all_chr,
    "mid": all_mid,
    "embedding_id": all_embed_id,
    "predicted_counts": all_preds,
    "target_counts": all_targets,
    "pred_total_density": all_pred_density,
    "obs_total_density": all_act_density,
})

output_df.to_csv(output_file, sep="\t", index=False)