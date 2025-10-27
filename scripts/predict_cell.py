import argparse
import numpy as np
import pandas as pd
import pyBigWig as pbw
import torch
import sys
from scipy.stats import pearsonr
from vinson.loss import poisson_loss
from genome_tools import GenomicInterval
from genome_tools.data.extractors import FastaExtractor
from vinson.interpretation import dinucleotide_shuffle, force_strict_ohe
from vinson.io import SamplesDensityExtractor
from vinson.models.sequence import BassetTrunkEmbed, CellEmbedding, EmbedModel

from vinson.vinson.utils.sequence_utils import intervals_to_ohe
from vinson.models.helpers import _Exp
import gzip

"""
This script loads a pretrained sequence + cell embedding model and applies it to
predict read counts at DHS peaks for a given assay group (ag). It:

1. Loads model weights, sequence embeddings, and read depth information.
2. Reads assay-specific peaks (BED) and normalized density tracks (bigWig).
3. Extracts centered genomic intervals, one-hot encodes sequences, and embeds
   the assay group vector.
4. Predicts read counts for each peak, normalizes them to assay read depth,
   and builds a predictions dataframe.
5. Computes performance metrics (Pearson correlation and mean Poisson NLL)
   between predicted and observed counts.
6. Outputs:
   - `<ag>.predictions.tsv`: per-peak predictions dataframe
   - `<ag>.metrics.txt`: summary metrics (Pearson, mean NLL)

Intended for batch processing across many assay groups, where outputs can be
later merged into a unified evaluation table.
"""


def main(args):
    # Create model
    embed = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk = BassetTrunkEmbed(embed.n_outputs)
    model = EmbedModel(trunk, embed, regression=True)
    #load model
    pretrained_state_dict = torch.load(args.checkpoint,map_location=torch.device("cpu"),)["state_dict"]
    model.load_state_dict(pretrained_state_dict)
    
    fasta_extr = FastaExtractor(args.fasta)
    embeddings_df = pd.read_csv(args.embeddings, sep="\t", index_col=0)
    cutcounts_df = pd.read_csv(args.read_depths, sep="\t")
    
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.exp = _Exp()
        def forward(self, seq, embed):
            return self.exp(self.model(seq, embed))
    model_wrapper = Wrapper(model).eval().to("cuda")

    #can change for argument input later use ag for ease of use
    ag = args.ag
    bed_file = f'{args.peaks_file_dir}/fdr0.001/{ag}.peaks.fdr0.001.bed.gz'
    bw_file  = f'{args.peaks_file_dir}/{ag}/{ag}.normalized_density.bw'

    def has_header(filepath):
        opener = gzip.open if filepath.endswith(".gz") else open
        with opener(filepath, "rt") as f:
            first_line = f.readline().strip().lower()
        return first_line.startswith("chr") or first_line.startswith("#chr")

    #some bed files have different header names and format
    if has_header(bed_file):
        bed_df = pd.read_csv(
            bed_file,
            sep="\t",
            usecols=[0, 1, 2],
            header=0,
            comment="#"
        )
        bed_df.columns = ["chrom", "start", "end"]
    else:
        bed_df = pd.read_csv(
            bed_file,
            sep="\t",
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
            header=None
        )
        
    fh = pbw.open(bw_file)

    # intervals + densities
    intervals, dens = [], []
    for _, row in bed_df.iterrows():
        try:
            mid = (int(row.start) + int(row.end)) // 2
            interval = GenomicInterval(row.chrom, mid, mid).widen(672)
            density  = fh.values(interval.chrom, mid, mid+1, numpy=True)[0]
            dens.append(density)
            intervals.append(interval)
        except Exception as e:
            print(f"[ERROR] ag={ag}, bed_file={bed_file}, chrom={row.chrom}, start={row.start}, end={row.end}, err={e}",
                  file=sys.stderr, flush=True)
            raise
    fh.close()

    # one-hot encode
    X_all = intervals_to_ohe(intervals, seqlen=1344, fasta_extr=fasta_extr)

    # embedding
    emb_vec = embeddings_df[ag].values
    X_embed = torch.tensor(emb_vec, dtype=torch.float32, device="cuda").unsqueeze(0)

    # predict in batches
    preds = []
    bs = 128
    with torch.no_grad():
        for i in range(0, len(X_all), bs):
            seq_batch = torch.as_tensor(X_all[i:i+bs], dtype=torch.float32, device="cuda")
            pred_batch = model_wrapper(seq_batch, X_embed).cpu().numpy()
            preds.append(pred_batch)
    preds = np.concatenate(preds).squeeze()

    # normalize
    read_depth = cutcounts_df.loc[cutcounts_df["ag_id"] == ag, "total_cutcounts"].values[0]
    pred_counts = (preds / 1e6) * read_depth
    target_counts = (np.array(dens) / 1e6) * read_depth

    # make df
    df = pd.DataFrame({
        "chrom": bed_df.chrom,
        "start": bed_df.start,
        "end": bed_df.end,
        "target_counts": target_counts,
        "pred_counts": pred_counts,
        "log_target_counts": np.log1p(target_counts),
        "log_pred_counts": np.log1p(pred_counts),
    })

    pearson_r, _ = pearsonr(df.log_target_counts, df.log_pred_counts)
    ps = 1e-6
    
    pred = torch.tensor(df.log_pred_counts, dtype=torch.float32)
    target = torch.tensor(df.log_target_counts, dtype=torch.float32)
    loss = poisson_loss(pred + ps, target + ps, reduction="none")
    mean_nll = loss.mean().item()

    # save outputs
    df.to_csv(f"{args.output_dir}/{ag}.predictions.tsv", sep="\t", index=False)
    with open(f"{args.output_dir}/{ag}.metrics.txt", "w") as f:
        f.write(f"{ag}\t{pearson_r:.4f}\t{mean_nll:.4f}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ag", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--read-depths", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--peak-file-dir", required=True)
    args = p.parse_args()
    main(args)
