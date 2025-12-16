#!/usr/bin/env python
import pandas as pd
import itertools
import os
import torch
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse
from vinson.loss import poisson_loss

"""
This script computes pairwise similarity metrics between assay groups (AGs) 
that share the same extended annotation. For each pair of AGs:

1. Uses bedtools to intersect their prediction files and identify overlapping peaks.
2. Loads the observed log target counts from both AGs for the shared peaks.
3. Computes two metrics:
   - Pearson correlation between log target counts
   - Mean Poisson negative log-likelihood (NLL) between log target counts
4. Stores both metrics for every AG pair.

Finally, the script summarizes results by writing out the mean pairwise Pearson 
and mean pairwise NLL values across all AG pairs for the given annotation.

Outputs:
- `<annotation>_summary.tsv`: table with annotation name, mean Pearson, mean NLL, and number of pairs
- Intermediate pairwise overlap files (`<ag1>_<ag2>_overlap.tsv`) are generated as needed

Run like:
    python /home/mbrannon/vinson/scripts/pairwisecor.py \
        --annotation $ANNOTATION
"""


ps = 1e-6

# Parse arguments
parser = argparse.ArgumentParser(description="Compute pairwise Pearson and NLL for one annotation")
parser.add_argument("--annotation", required=True, help="Embedding annotation to process")
outdir = '/home/mbrannon/tmp/pairwise_rep_nll/'
args = parser.parse_args()

bedtools_path = "/home/mbrannon/.local/miniconda3/envs/biotools/bin/bedtools"

# Load metadata
meta = pd.read_csv('/home/mbrannon/tmp/ag_metadata.tsv', sep="\t")

# Filter for this annotation
group = meta[meta["extended_annotation"] == args.annotation]
ag_ids = group["ag_id"].tolist()

pearsons, nlls = [], []
pairs = list(itertools.combinations(ag_ids, 2))

for ag1, ag2 in tqdm(pairs, desc=f"{args.annotation} pairs"):
    overlap_file = f'/home/mbrannon/tmp/ag_preds/{ag1}_{ag2}_overlap.tsv'
    if not os.path.exists(overlap_file):
        os.system(
            f"{bedtools_path} intersect "
            f"-a /home/mbrannon/tmp/ag_preds/{ag1}.predictions.tsv "
            f"-b /home/mbrannon/tmp/ag_preds/{ag2}.predictions.tsv "
            f"-wa -wb -f 0.5 > {overlap_file}"
        )
    # Load overlap results
    overlap_df = pd.read_csv(overlap_file, sep="\t", header=None)
    overlap_df.columns = [
        'chrom1','start1','end1','target_counts1','pred_counts1','log_target_counts1','log_pred_counts1',
        'chrom2','start2','end2','target_counts2','pred_counts2','log_target_counts2','log_pred_counts2'
    ]
    
    # Pearson correlation
    pearson_r, _ = pearsonr(overlap_df['log_target_counts1'], overlap_df['log_target_counts2'])
    pearsons.append(pearson_r)
    
    # Poisson loss (mean NLL)
    pred = torch.tensor(overlap_df['log_target_counts1'].values, dtype=torch.float32)
    target = torch.tensor(overlap_df['log_target_counts2'].values, dtype=torch.float32)
    loss = poisson_loss(pred + ps, target + ps, reduction="none")
    mean_nll = loss.mean().item()
    nlls.append(mean_nll)

# Compute means
mean_pearson = sum(pearsons)/len(pearsons) if pearsons else float('nan')
mean_nll = sum(nlls)/len(nlls) if nlls else float('nan')

# Save result
out_file = os.path.join(outdir, f"{args.annotation}_summary.tsv")
pd.DataFrame([{
    "embedding_annotation": args.annotation,
    "mean_pairwise_pearson": mean_pearson,
    "mean_pairwise_nll": mean_nll,
    "n_pairs": len(pairs)
}]).to_csv(out_file, sep="\t", index=False)

print(f"Finished {args.annotation}: "
      f"mean_pearson={mean_pearson:.4f}, mean_nll={mean_nll:.4f}, pairs={len(pairs)}")
