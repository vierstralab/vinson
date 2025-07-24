import torch
from torch.utils.data import DataLoader
import numpy as np
from genome_tools import GenomicInterval as genomic_interval
import pandas as pd
import argparse
from vinson.dataset import SequenceEmbeddingDataset
from vinson.model import (
    BassetTrunkEmbed,
    CellEmbedding,
    VinsonEmbedModel,
)

#inputs
parser = argparse.ArgumentParser(description="Run prediction with Vinson model.")
parser.add_argument("--eval-samples", required=True, help="Path to evaluation samples HDF5 file")
parser.add_argument("--embeddings", required=True, help="Path to embeddings TSV file")
parser.add_argument("--read-depths", required=True, help="Path to total cut counts file")
parser.add_argument("--fasta", required=True, help="Path to FASTA file")
parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
parser.add_argument("--output", required=True, help="Path to output TSV file")
args = parser.parse_args()

eval_samples_file = args.eval_samples
embeddings_file = args.embeddings
read_depth_file = args.read_depths
fasta_file = args.fasta
model_ckpt = args.checkpoint
output_file = args.output

#dataset
dataset = SequenceEmbeddingDataset(
    eval_samples_file,
    embeddings_file,
    read_depth_file,
    fasta_file,
    reverse_complement=True,
    jitter=0,
    noise=0,
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,  # important: preserve order
    num_workers=1,
    pin_memory=True,
    drop_last=False,
)

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed = CellEmbedding(n_inputs=637, n_layers=1)
trunk = BassetTrunkEmbed(embed)
model = VinsonEmbedModel(trunk, regression=True)

pretrained_state_dict = torch.load(
    model_ckpt,
    map_location=torch.device("cpu"),
)["state_dict"]

model.load_state_dict(pretrained_state_dict)
model.to(device)
model.eval()


# predict with metadata
all_preds = []
all_targets = []
all_chr = []
all_mid = []
all_embed_id = []


#store sample id
#store some sort of chr marker for dhs
#no grad means dont have to detach everytime
with torch.no_grad():
    for batch in dataloader:
        X_seq = batch["seq"].to(device)
        X_embed = batch["embed"].to(device)
        density = batch["density"].to(device)
        read_depth = batch["read_depth"].to(device)
        chrom = batch["chrom"] 
        mid = batch["mid"]      
        embed_id = batch["sample_id"] 

        density = density + 0.001
        y_pred = model(X_seq, X_embed).squeeze()

        pred_counts = (torch.exp(y_pred) / 1e6 * read_depth) + 1.0
        target_counts = (density / 1e6 * read_depth) + 1.0

        # Store results + metadata
        all_preds.append(pred_counts.cpu().numpy())
        all_targets.append(target_counts.cpu().numpy())
        all_chr.extend(chrom)  # Already CPU
        all_mid.extend(mid)
        all_embed_id.extend(embed_id)  # Already CPU

# === Combine all into DataFrame ===
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

output_df = pd.DataFrame({
    "chrom": all_chr,
    "mid":all_mid,
    "embedding_id": all_embed_id,
    "predicted_counts": all_preds,
    "target_counts": all_targets,
})

output_df.to_csv(output_file, sep="\t", index=False)