import torch
from torch.utils.data import DataLoader
import numpy as np
from genome_tools import GenomicInterval as genomic_interval
import pandas as pd
import argparse
from tqdm import tqdm

from vinson.dataset import SeqEmbedDataset
from vinson.loss import poisson_loss
from vinson.model import BassetTrunkEmbed, CellEmbedding, EmbedModel


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
    dataset_kwargs["negative_samples_rate"] = 1

dataset = SeqEmbedDataset(
    samples_file,
    embeddings_file,
    read_depth_file,
    fasta_file,
    **dataset_kwargs,
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last=True,
)
#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
embed = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
trunk = BassetTrunkEmbed(embed.n_outputs)
model = EmbedModel(trunk, embed, regression=True)

pretrained_state_dict = torch.load(
    model_ckpt,
    map_location=torch.device("cpu"),
)["state_dict"]

model.load_state_dict(pretrained_state_dict)
# model.to(device)
# model.eval()


# predict with metadata
all_preds = []
all_targets = []
all_chr = []
all_mid = []
all_embed_id = []
all_pred_density = []


#store sample id
#store some sort of chr marker for dhs
#no grad means dont have to detach everytime
with torch.no_grad():
    for batch in dataloader:
        X_seq = batch["seq"].to(device)
        X_embed = batch["embed"].to(device)
        density = batch["density"].to(device)
        read_depth = batch["read_depth"].to(device)
        bg = batch['bg'].to(device)
        chrom = batch["chrom"] 
        mid = batch["mid"]      
        embed_id = batch["sample_id"] 
        
        y_pred = model(X_seq, X_embed).squeeze()
        pred_counts = (torch.exp(y_pred) / 1e6 * read_depth) + bg
        target_counts = density / 1e6 * read_depth
        bg_density = bg / read_depth * 1e6
        pred_total_density = torch.exp(y_pred) + bg_density

        # Store results + metadata
        all_preds.append(pred_counts.cpu().numpy())
        all_targets.append(target_counts.cpu().numpy())
        all_chr.extend(chrom)  # Already CPU
        all_mid.extend(mid)
        all_embed_id.extend(embed_id)  # Already CPU
        all_pred_density.append(pred_total_density.cpu().numpy())

# === Combine all into DataFrame ===
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
all_density = np.concatenate(all_pred_density)

output_df = pd.DataFrame({
    "chrom": all_chr,
    "mid":all_mid,
    "embedding_id": all_embed_id,
    "predicted_counts": all_preds,
    "target_counts": all_targets,
    "pred_total_density":all_density,
})

output_df.to_csv(output_file, sep="\t", index=False)