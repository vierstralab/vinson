import torch
from torch.utils.data import DataLoader
import numpy as np
from genome_tools import GenomicInterval as genomic_interval
import pandas as pd
import argparse
from vinson.loss import binomial_mixture_loss, binomial_mixture_nll, binomial_mixture_normed_loss
from vinson.dataset import VariantEmbedDataset
from vinson.model import (
    CellEmbedding,
    BassetTrunkEmbed,  
    VariantEmbedModel, 
    VariantEmbedModelWrapper)
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
dataset = VariantEmbedDataset(
    eval_samples_file,
    embeddings_file,
    fasta_file,
    reverse_complement=False,
    jitter=0,
    noise=0,
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=True,
)



#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
embed = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
trunk = BassetTrunkEmbed(embed.n_outputs)
model = VariantEmbedModel(trunk, embed)


pretrained_state_dict = torch.load(
    model_ckpt,
    map_location=torch.device("cpu"),
)["state_dict"]

model_dict = model.state_dict()
model_dict.update(pretrained_state_dict)
model.load_state_dict(model_dict, strict=False)


# predict with metadata
all_preds = []
all_targets = []
all_chr = []
all_pos = []
all_ids = []
all_ref = []
all_alt = []

#store sample id
#store some sort of chr marker for dhs
#no grad means dont have to detach everytime
with torch.no_grad():
    for batch in dataloader:
        chrom = batch["chrom"]            # list or tensor of chroms
        pos = batch["pos"]                # list or tensor of positions
        sample_id = batch["sample_id"]    # list of sample IDs
        X_seq_ref = batch["seq_ref"].to(device)
        X_seq_alt = batch["seq_alt"].to(device)
        X_embed = batch["embed"].to(device)
        ref_counts = batch["ref_counts"]
        total_counts = batch["total_counts"]
        bad = batch["bad_score"]
        lfc = batch["lfc"]
        ref = batch['ref']
        alt = batch['alt']

        y = model(X_seq_ref, X_seq_alt, X_embed).squeeze()

        # Append to list
        all_preds.append(y.cpu().numpy())
        all_targets.append(lfc.cpu().numpy())

        # Make sure to flatten and extend
        all_chr.extend(chrom)        # assume list of strings
        all_pos.extend(pos.tolist())          # assume list or tensor of ints
        all_ids.extend(sample_id)    # assume list of strings or IDs
        all_alt.extend(alt)
        all_ref.extend(ref)


# === Combine all into DataFrame ===
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

output_df = pd.DataFrame({
    "sample_id": all_ids,
    "chr": all_chr,
    "pos": all_pos,
    "ref": all_ref,
    "alt": all_alt,
    "predicted_lfc": all_preds,
    "target_lfc": all_targets,
    
})

output_df.to_csv(output_file, sep="\t", index=False)