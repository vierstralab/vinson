import os
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from genome_tools.data.anndata import read_zarr_backed
import scipy.sparse as sp
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from collections import defaultdict
import gc
# -------------------------
# Paths
# -------------------------
# adata_file = "/net/seq/data2/projects/mbrannon/6_2_both.h5ad"
# -------------------------
# Model paths
# -------------------------
adata_file = "/home/mbrannon/tmp/adata_allcombo.h5ad"
zarr_anndata = adata_file

fasta_file = "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
model_config = "/net/seq/data2/projects/mbrannon/variant_model_paper/2026_06_08_happy_bose/run_config.yaml"
genotype_file = "/net/seq/data2/projects/mbrannon/variant_model_paper/genotypes/phased_genotype_dnase_and_atac.bed.gz"
checkpoint = "/net/seq/data2/projects/mbrannon/variant_model_paper/2026_06_08_happy_bose/checkpoints/epoch=14-step=58021-val_loss=3.44.ckpt"

h5_out_dir = "/home/mbrannon/tmp/7_12_prediction_inputs"
os.makedirs(h5_out_dir, exist_ok=True)

adata = ad.read_h5ad(
    adata_file,
    backed="r"
)

# -------------------------
# Settings
# -------------------------
variant_chunk_size = 10000
def get_layer_values(layer, obs_idx, var_idx):
    x = layer[obs_idx][:, var_idx]
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32).ravel()

# -------------------------
# Load AnnData
# -------------------------
print("Loading AnnData...")
# adata = ad.read_h5ad(adata_file, backed="r")
# -------------------------
# Prepare variants
# -------------------------
print("Preparing variants...")

variants = adata.var.copy()

variants["#chr"] = variants["#chr"].astype(str)
variants["end"] = variants["end"].astype(int)
variants["ref"] = variants["ref"].astype(str)
variants["alt"] = variants["alt"].astype(str)

variants = variants[
    [
        "#chr",
        "end",
        "ref",
        "alt",
    ]
]

print(f"{len(variants):,} variants")

# -------------------------
# Prepare samples
# -------------------------
print("Preparing samples...")

samples = adata.obs.copy()

samples["sample_id"] = adata.obs_names.astype(str)

samples["indiv_id"] = (
    adata.obsm["indiv_id"]
    .ravel()
    .astype(str)
)

samples["group_id"] = samples["extended_annotation"].astype(str)

samples["obs_idx"] = np.arange(adata.n_obs)

samples = samples.reset_index(drop=True)

print(f"{len(samples):,} samples")
print(f"{samples['group_id'].nunique()} groups")

groups = sorted(samples["group_id"].unique())

# -------------------------
# Write H5 files
# -------------------------
prediction_manifest = []

for group_id in groups:
    if pd.isna(group_id) or str(group_id).lower() == "nan":
        continue

    print(f"\nProcessing {group_id}")

    group_samples = samples.loc[
        samples["group_id"] == group_id,
        ["sample_id", "indiv_id", "obs_idx"]
    ]
    safe_group_id = (
            group_id
            .replace("/", "_")
            .replace(" ", "_")
        )

    n_samples = len(group_samples)

    obs_idx = group_samples["obs_idx"].to_numpy()

    for start in tqdm(
        range(0, len(variants), variant_chunk_size),
        desc=group_id,
    ):

        var_chunk = variants.iloc[start:start + variant_chunk_size]

        n_variants = len(var_chunk)

        outfile = os.path.join(
            h5_out_dir,
            f"{safe_group_id}_variants_{start}_{start+n_variants}.h5",
        )


        if os.path.exists(outfile):
            prediction_manifest.append(
                {
                    "prefix": safe_group_id,
                    "zarr_anndata": zarr_anndata,
                    "fasta_file": fasta_file,
                    "model_config": model_config,
                    "genotype_file": genotype_file,
                    "checkpoint": checkpoint,
                    "dhs_dataset": outfile,
                }
            )
            continue

        var_idx = np.arange(start, start + n_variants)

        # -------------------------
        # Extract layer values
        # -------------------------

        ref_counts = get_layer_values(
            adata.layers["ref_counts"],
            obs_idx,
            var_idx
        )

        alt_counts = get_layer_values(
            adata.layers["alt_counts"],
            obs_idx,
            var_idx
        )

        total_counts = get_layer_values(
            adata.layers["total_counts"],
            obs_idx,
            var_idx
        )

        bad = get_layer_values(
            adata.layers["BAD"],
            obs_idx,
            var_idx
        )

        logit_es = get_layer_values(
            adata.layers["logit_es"],
            obs_idx,
            var_idx
        )


        # -------------------------
        # Cartesian product
        # -------------------------

        sample_rep = np.repeat(
            group_samples["sample_id"].values,
            n_variants,
        )

        indiv_rep = np.repeat(
            group_samples["indiv_id"].values,
            n_variants,
        )

        chr_rep = np.tile(
            var_chunk["#chr"].values,
            n_samples,
        )

        pos_rep = np.tile(
            var_chunk["end"].values,
            n_samples,
        )

        ref_rep = np.tile(
            var_chunk["ref"].values,
            n_samples,
        )

        alt_rep = np.tile(
            var_chunk["alt"].values,
            n_samples,
        )


        

        assert len(sample_rep) == len(ref_counts)
        assert len(sample_rep) == len(chr_rep)
        assert len(sample_rep) == len(logit_es)
        with h5py.File(outfile, "w") as f:

            string_dtype = h5py.string_dtype("utf-8")

            f.create_dataset(
                "chrom",
                data=chr_rep,
                dtype=string_dtype,
                compression="gzip",
            )

            f.create_dataset(
                "pos",
                data=pos_rep,
                compression="gzip",
            )

            f.create_dataset(
                "ref",
                data=ref_rep,
                dtype=string_dtype,
                compression="gzip",
            )

            f.create_dataset(
                "alt",
                data=alt_rep,
                dtype=string_dtype,
                compression="gzip",
            )

            f.create_dataset(
                "sample_id",
                data=sample_rep,
                dtype=string_dtype,
                compression="gzip",
            )

            f.create_dataset(
                "indiv_id",
                data=indiv_rep,
                dtype=string_dtype,
                compression="gzip",
            )

            f.create_dataset(
                "ref_counts",
                data=ref_counts,
                compression="gzip",
            )

            f.create_dataset(
                "alt_counts",
                data=alt_counts,
                compression="gzip",
            )

            f.create_dataset(
                "total_counts",
                data=total_counts,
                compression="gzip",
            )

            f.create_dataset(
                "BAD",
                data=bad,
                compression="gzip",
            )

            f.create_dataset(
                "logit_es",
                data=logit_es,
                compression="gzip",
            )


        prediction_manifest.append(
            {
                "prefix": safe_group_id,
                "zarr_anndata": zarr_anndata,
                "fasta_file": fasta_file,
                "model_config": model_config,
                "genotype_file": genotype_file,
                "checkpoint": checkpoint,
                "dhs_dataset": outfile,
            }
        )

# -------------------------
# Save manifest
# -------------------------
# =====================================================
# Save manifest
# =====================================================

manifest_df = pd.DataFrame(prediction_manifest)

manifest_out = os.path.join(
    h5_out_dir,
    "prediction_input_manifest.tsv",
)

manifest_df.to_csv(
    manifest_out,
    sep="\t",
    index=False,
)

print()
print(f"Finished writing {len(prediction_manifest):,} H5 files")
print(f"Manifest saved to:\n{manifest_out}")