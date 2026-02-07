#!/usr/bin/env python

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import gc
from tqdm import tqdm

#-----------------------------------------
# Paths
#-----------------------------------------
CAVS_PATH = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-cavs.v5/output/differential_pvals.extended_annotation.bed"
NONAGG_PATH = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-cavs.v5/output/non_aggregated.all.parquet"
META_PATH = "/net/seq/data2/projects/sabramov/ENCODE4/dnase-cavs.v5/meta+sample_ids.tsv"

OUT_PARQUET = "/net/seq/data2/projects/mbrannon/nonagg_merged_stats.parquet"

#-----------------------------------------
# Load CAVS
#-----------------------------------------
print("Loading CAVS annotations…")
cavs = pd.read_table(CAVS_PATH)

merge_keys = ['#chr', 'start', 'end', 'ref', 'alt', 'ID']
cavs = cavs.drop_duplicates(subset=merge_keys)

bool_cols = ['cell_selective', 'overall_imbalanced', 'concordant', 'discordant']
for c in bool_cols:
    if c in cavs.columns:
        cavs[c] = cavs[c].fillna(False).astype(bool)

gc.collect()

#-----------------------------------------
# Individual map
#-----------------------------------------
print("Loading individual mapping…")
indiv_map = pd.read_table(META_PATH).set_index("sample_id")["indiv_id"]

#-----------------------------------------
# Arrow dataset
#-----------------------------------------
print("Processing non-aggregated variants in batches…")
dataset = ds.dataset(NONAGG_PATH, format="parquet")

batch_size = 2_000_000
writer = None

for batch in tqdm(dataset.to_batches(batch_size=batch_size), desc="Batches"):
    nonagg = batch.to_pandas()

    # Map indiv_id
    nonagg["indiv_id"] = nonagg["sample_id"].map(indiv_map)

    # Merge CAVS
    merged = nonagg.merge(
        cavs,
        on=merge_keys,
        how="left",
        suffixes=("", "_cavs")
    )

    #-----------------------------------------
    # FORCE BOOLEAN TYPES AFTER MERGE (CRITICAL)
    #-----------------------------------------
    for c in bool_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(False).astype(bool)
        else:
            merged[c] = False

    #-----------------------------------------
    # Variant-level statistics
    #-----------------------------------------
    if "variant_id" in merged.columns:
        merged["group_es_var"] = merged["group_es_var"].fillna(0.0)
        merged["mean_group_mse"] = merged["mean_group_mse"].fillna(0.0)

        merged["full_group_std"] = np.sqrt(
            merged["group_es_var"] + merged["mean_group_mse"]
        )

        merged.loc[merged["full_group_std"] == 0, "full_group_std"] = np.nan

        merged["z"] = (merged["group_es"] - 0.5) / merged["full_group_std"]

        merged["fdr_group"] = merged["fdr_group"].fillna(1.0)
        merged["signif_z"] = np.where(
            merged["fdr_group"] < 0.1,
            merged["z"],
            0.0
        )

        g = merged.groupby("variant_id", sort=False)

        merged["max_z"] = g["signif_z"].transform("max")
        merged["min_z"] = g["signif_z"].transform("min")

        denom = np.sqrt(
            g["group_es_var"].transform("max") +
            g["group_es_var"].transform("min")
        )
        denom = denom.replace(0, np.nan)

        merged["delta_es_q"] = (
            g["group_es"].transform("max") -
            g["group_es"].transform("min")
        ) / denom

        merged["discordance_score"] = (
            merged["min_z"] *
            (((merged["max_z"] * merged["min_z"]) < 0) * 2 - 1)
        )

    #-----------------------------------------
    # Category logic (now SAFE)
    #-----------------------------------------
    merged["category"] = np.select(
        [
            merged["discordant"] & merged["overall_imbalanced"],
            merged["discordant"] & ~merged["overall_imbalanced"],
            merged["concordant"] & merged["overall_imbalanced"],
            merged["concordant"] & ~merged["overall_imbalanced"],
            merged["cell_selective"],
            merged["overall_imbalanced"],
            ~merged["overall_imbalanced"],
        ],
        [
            "discordant_and_overall",
            "discordant",
            "concordant_and_overall",
            "concordant",
            "weak_cell_selective",
            "not_cell_selective",
            "not_imbalanced",
        ],
        default="wtf",
    )

    #-----------------------------------------
    # Coverage filter
    #-----------------------------------------
    if "coverage" in merged.columns:
        merged = merged.loc[merged["coverage"] >= 35]

    #-----------------------------------------
    # Incremental parquet write (CORRECT)
    #-----------------------------------------
    table = pa.Table.from_pandas(merged, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            OUT_PARQUET,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)

    del nonagg, merged, table
    gc.collect()

if writer is not None:
    writer.close()

print("DONE")
print(f"Wrote: {OUT_PARQUET}")
