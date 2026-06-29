#!/usr/bin/env python

import anndata as ad
from genome_tools.data.anndata import read_zarr_backed
import pandas as pd
import scipy.sparse as sp
import numpy as np
import gc

# =========================================================
# Paths
# =========================================================

PARQUET_PATH = "/home/mbrannon/tmp/5_7_both_atac_dnase_variants_phased_passedgeno.parquet"
DHS_PATH = "/net/seq/data2/projects/ENCODE4Plus/REGULOME/one_big_beautiful_index/latest.reference_anndata.zarr"
ATAC_PATH = "/net/seq/data2/projects/ENCODE4Plus/REGULOME/one_big_beautiful_index/latest.SRA_anndata.zarr"

OUT_DIR = "/net/seq/data2/projects/mbrannon/"

# =========================================================
# Load RAW data (IMPORTANT: no filtering here)
# =========================================================

used_results_raw = pd.read_parquet(PARQUET_PATH)

dhsadata = read_zarr_backed(DHS_PATH)
atac_anndata = read_zarr_backed(ATAC_PATH)

# =========================================================
# Column setup
# =========================================================

layer_cols_numeric = [
    "ref_counts",
    "alt_counts",
    "BAD",
    "logit_es",
    "FDR_sample",
    "inverse_mse",
    "total_counts",
]

used_results_raw[layer_cols_numeric] = used_results_raw[layer_cols_numeric].astype(float)

# =========================================================
# Variant ID
# =========================================================

used_results_raw["variant_id"] = (
    used_results_raw["#chr"]
    + ":"
    + used_results_raw["end"].astype(str)
    + ":"
    + used_results_raw["ref"]
    + ":"
    + used_results_raw["alt"]
)

used_results_raw["calc_alt_counts"] = (
    used_results_raw["total_counts"] - used_results_raw["ref_counts"]
)

# =========================================================
# Base masks (global, reused)
# =========================================================

passed = (
    (used_results_raw["total_counts"] > 20)
    & (used_results_raw["ref_counts"] > 0)
    & (used_results_raw["calc_alt_counts"] > 0)
)

pos_mask = passed & (used_results_raw["FDR_sample"] < 0.1) & (used_results_raw["logit_es"].abs() < 6)
neg_mask = passed & (used_results_raw["FDR_sample"] > 0.5)

pos_df = used_results_raw[pos_mask]
neg_df = used_results_raw[neg_mask]

print("Pos:", len(pos_df), "Neg:", len(neg_df))

# =========================================================
# Split datasets (DO NOT MODIFY RAW)
# =========================================================

atac_df = used_results_raw.query(
    'indiv_id.str.startswith("ATAC")',
    engine="python"
)

dnase_df = used_results_raw.query(
    'indiv_id.str.startswith("DNASE")',
    engine="python"
)

both_df = used_results_raw

# atac_bad15 = atac_df.query("BAD <= 1.5")
# dnase_bad15 = dnase_df.query("BAD <= 1.5")
both_bad15 = both_df.query("BAD <= 1.5")

# =========================================================
# OBS (shared across ALL datasets)
# =========================================================

ref_obs = dhsadata.obs.copy()
atac_obs = atac_anndata.obs.copy()

ref_obs.index = ref_obs.index.astype(str)
atac_obs.index = atac_obs.index.astype(str)

obs = pd.concat([ref_obs, atac_obs])

obs["modality"] = np.where(
    obs.index.str.startswith("AG"),
    "DNASE",
    "ATAC",
)
# =========================================================
# VAR (shared)
# =========================================================

var_df = (
    pd.concat([pos_df, neg_df])
    .drop_duplicates("variant_id")
    .set_index("variant_id")[
        ["#chr", "start", "end", "ID", "ref", "alt", "AAF", "RAF"]
    ]
)

# =========================================================
# GLOBAL validation split (500 samples total)
# =========================================================

n_val = 500

n_atac_val = int(round((obs["modality"] == "ATAC").mean() * n_val))
n_dnase_val = n_val - n_atac_val

val_atac = (
    obs.query('modality == "ATAC"')
    .sample(n=n_atac_val, random_state=42)
    .index
)

val_dnase = (
    obs.query('modality == "DNASE"')
    .sample(n=n_dnase_val, random_state=42)
    .index
)

val_samples = np.concatenate([val_atac, val_dnase])

# =========================================================
# Indexers
# =========================================================

obs_indexer = pd.Series(range(len(obs)), index=obs.index)
var_indexer = pd.Series(range(len(var_df)), index=var_df.index)

# =========================================================
# Embeddings (shared)
# =========================================================

ref_emb = pd.DataFrame(
    dhsadata.obsm["motif_embeddings"],
    index=dhsadata.obs_names
)

atac_emb = pd.DataFrame(
    atac_anndata.obsm["motif_embeddings"],
    index=atac_anndata.obs_names
)

ref_emb.index = ref_emb.index.astype(str)
atac_emb.index = atac_emb.index.astype(str)

combined_emb = pd.concat([ref_emb, atac_emb]).reindex(obs.index)

del ref_emb
del atac_emb
del dhsadata
del atac_anndata
gc.collect()

# =========================================================
# Builder
# =========================================================

def build_dataset(df, name, out_path):

    print(f"\n================ {name} ================")

    # =====================================================
    # Dataset-specific positives / negatives
    # =====================================================

    df = df.copy()

    df["calc_alt_counts"] = (
        df["total_counts"] - df["ref_counts"]
    )

    passed_local = (
        (df["total_counts"] > 20)
        & (df["ref_counts"] > 0)
        & (df["calc_alt_counts"] > 0)
    )

    pos_df_local = df[
        passed_local
        & (df["FDR_sample"] < 0.1)
        & (df["logit_es"].abs() < 6)
    ]

    neg_df_local = df[
        passed_local
        & (df["FDR_sample"] > 0.5)
    ]

    print("Positives:", len(pos_df_local))
    print("Negatives:", len(neg_df_local))

    if len(pos_df_local) == 0:
        raise ValueError(f"{name}: no positive examples")

    if len(neg_df_local) < len(pos_df_local):
        raise ValueError(
            f"{name}: negatives ({len(neg_df_local)}) "
            f"< positives ({len(pos_df_local)})"
        )

    # =====================================================
    # Create AnnData
    # =====================================================

    adata = ad.AnnData(
        X=None,
        obs=obs.copy(),
        var=var_df.copy()
    )

    obs_idx = pd.Series(
        range(len(adata.obs)),
        index=adata.obs_names
    )

    var_idx = pd.Series(
        range(len(adata.var)),
        index=adata.var_names
    )

    adata.obsm["motif_embeddings"] = combined_emb

    epochs = ["epoch_1", "epoch_2", "epoch_3"]

    # =====================================================
    # Build sparse layers
    # =====================================================

    for i, epoch in enumerate(epochs):

        print(f"\n========== {epoch} ==========")

        neg_sampled = neg_df_local.sample(
            n=len(pos_df_local),
            random_state=100 + i
        )

        epoch_df = pd.concat(
            [pos_df_local, neg_sampled],
            axis=0
        ).sample(
            frac=1,
            random_state=100 + i
        )

        row_idx = epoch_df["sample_id"].map(obs_idx).to_numpy()
        col_idx = epoch_df["variant_id"].map(var_idx).to_numpy()

        missing_rows = np.isnan(row_idx).sum()
        missing_cols = np.isnan(col_idx).sum()

        print("Missing rows:", missing_rows)
        print("Missing cols:", missing_cols)

        keep = (
            ~np.isnan(row_idx)
            & ~np.isnan(col_idx)
        )

        row_idx = row_idx[keep].astype(np.int32)
        col_idx = col_idx[keep].astype(np.int32)

        epoch_df = epoch_df.iloc[keep]

        for col in layer_cols_numeric:

            print(f"[BUILD] {col}.{epoch}")

            vals = epoch_df[col].to_numpy()

            mat = sp.coo_matrix(
                (vals, (row_idx, col_idx)),
                shape=adata.shape
            ).tocsr()

            adata.layers[f"{col}.{epoch}"] = mat

        del epoch_df
        del neg_sampled
        gc.collect()

    # =====================================================
    # metadata
    # =====================================================

    mapping = (
        used_results_raw[
            ["sample_id", "indiv_id"]
        ]
        .dropna()
        .drop_duplicates("sample_id")
        .set_index("sample_id")["indiv_id"]
    )

    adata.obsm["indiv_id"] = (
        adata.obs_names
        .map(mapping)
        .fillna("")
        .to_numpy()
        .reshape(-1, 1)
    )

    # =====================================================
    # sample split
    # =====================================================

    if name.startswith("atac"):
        dataset_val_samples = val_atac

    elif name.startswith("dnase"):
        dataset_val_samples = val_dnase

    else:
        dataset_val_samples = val_samples

    adata.obsm["split_data"] = np.where(
        adata.obs_names.isin(dataset_val_samples),
        "val",
        "train"
    )
    

    # =====================================================
    # chromosome split
    # =====================================================

    validation_chroms = ["chr9", "chr22"]

    adata.varm["split_data"] = np.where(
        adata.var["#chr"].isin(validation_chroms),
        "val",
        "train"
    )

    # =====================================================
    # metadata
    # =====================================================

    

    adata.uns["val_samples"] = val_samples.tolist()
    adata.uns["epoch_names"] = epochs

    # =====================================================
    # save
    # =====================================================

    print("Saving:", out_path)

    adata.write_h5ad(
        out_path,
        compression="gzip"
    )

    del adata
    gc.collect()


# =========================================================
# RUN ALL 6 DATASETS
# =========================================================

datasets = [
    # ("atac", atac_df),
    # ("dnase", dnase_df),
    ("both", both_df),

    # ("atac_bad15", atac_bad15),
    # ("dnase_bad15", dnase_bad15),
    ("both_bad15", both_bad15),
]

for name, df in datasets:
    build_dataset(df, name, OUT_DIR + f"6_5_nonphased_{name}.h5ad")

print("\nDONE")
