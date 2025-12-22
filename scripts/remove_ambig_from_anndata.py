#!/usr/bin/env python3
import anndata as ad
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict
from genome_tools.data.anndata import read_zarr_backed
from genome_tools import GenomicInterval, VariantInterval, df_to_variant_intervals
from genome_tools.data.extractors import TabixExtractor
import numpy as np
import os

# ------------------------------
# Load data
# ------------------------------
print("Loading DHS and variant data...")
train_dhs_adata = ad.read_h5ad(
    '/home/sabramov/public_html/collaborations/vinson/NOV03/NOV03_3_epochs_with_weights_fc_squared_50.h5ad'
)
adata = read_zarr_backed(
    '/net/seq/data2/projects/ENCODE4Plus/REGULOME/one_big_beautiful_index/latest.reference_anndata.zarr'
)

cols = [
    "#chr", "start", "end",
    "ref", "alt",
    "sample_id",
    "ref_counts", "alt_counts",
    "coverage", "BAD", "logit_es",
    "ID", "AAF", "RAF"
]

non_aggregated_variants = pd.read_parquet(
    "/net/seq/data2/projects/sabramov/ENCODE4/dnase-cavs.v5/output/non_aggregated.all.parquet",
    columns=cols
)
non_aggregated_variants['variant_id'] = (
    non_aggregated_variants['#chr'] + ":" +
    non_aggregated_variants['end'].astype(str) + ':' +
    non_aggregated_variants['ref'] + ':' +
    non_aggregated_variants['alt']
)
non_aggregated_variants['total_counts'] = non_aggregated_variants['ref_counts'] + non_aggregated_variants['alt_counts']

train_df = non_aggregated_variants.query("coverage >= 35").copy()
var_df = train_df.drop_duplicates('variant_id').set_index('variant_id')[['#chr', 'start', 'end', 'ID', 'ref', 'alt', 'AAF', 'RAF']]
train_var_adata = ad.AnnData(X=None, var=var_df, obs=adata.obs)

row_idx = train_var_adata.obs_names.get_indexer(train_df["sample_id"])
col_idx = train_var_adata.var_names.get_indexer(train_df["variant_id"])

for column in ['ref_counts', 'total_counts', 'BAD', 'logit_es']:
    data = sp.coo_matrix(
        (train_df[column], (row_idx, col_idx)),
        shape=train_var_adata.shape
    ).tocsr()
    train_var_adata.layers[column] = data

indiv_id_mapping = pd.read_table(
    '/net/seq/data2/projects/sabramov/ENCODE4/dnase-cavs.v5/meta+sample_ids.tsv'
).set_index('sample_id')

train_var_adata.obsm['indiv_id'] = train_var_adata.obs_names.map(
    indiv_id_mapping['indiv_id']
).fillna('').to_numpy().astype(str)
indiv_ids = train_var_adata.obsm['indiv_id']

genotype_file = '/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/phasing/output/all_phased.bed.gz'
genotype_extr = TabixExtractor(
    genotype_file,
    skiprows=1,
    columns=[
        "chrom", "start", "end", "ref", "alt", "indiv_id", "gt", "phase_block"
    ],
    na_values={"phase_block": "."}
)

# ------------------------------
# Scan for ambiguous variants
# ------------------------------
ambiguous_examples = set()
print("Scanning variant examples...")
total = len(row_idx)

for r, c in tqdm(zip(row_idx, col_idx), total=total, desc="Scanning variants"):
    indiv_id = indiv_ids[r]
    if pd.isna(indiv_id) or indiv_id in ("None", ""):
        continue

    loc = train_var_adata.var.iloc[c]
    chrom, pos, ref, alt = loc["#chr"], loc["end"], loc["ref"], loc["alt"]
    interval = GenomicInterval(chrom, pos, pos).widen(1354 // 2)

    try:
        variants = genotype_extr[interval]
    except ValueError:
        continue

    # Filter to this individual
    variants = variants[variants["indiv_id"] == indiv_id]
    if variants.empty:
        continue

    # Ensure phased info exists
    if "phase_set" not in variants.columns:
        if "phase_block" in variants.columns:
            variants = variants.rename(columns={"phase_block": "phase_set"})
        else:
            variants["phase_set"] = None

    # Convert to VariantInterval objects
    variants = df_to_variant_intervals(variants, extra_columns=("gt", "phase_set"))

    # Detect ambiguous positions: same start, same indiv, different ref/alt
    starts = defaultdict(list)
    for v in variants:
        starts[v.start].append(v)

    ambiguous = False
    for pos_in_interval, vars_at_pos in starts.items():
        unique_variants = {(v.ref, v.alt) for v in vars_at_pos}
        if len(unique_variants) > 1:
            ambiguous = True
            print(f"AMBIGUOUS at row {r}, col {c}, indiv={indiv_id}, pos={pos_in_interval}", flush=True)
            for v in vars_at_pos:
                print(f"  {v.chrom}:{v.start}-{v.end} {v.ref}>{v.alt} gt={getattr(v, 'gt', None)}", flush=True)
            break  # Stop after first ambiguous position

    if ambiguous:
        ambiguous_examples.add((r, c))
        continue

# ------------------------------
# Save final ambiguous examples
# ------------------------------
ambiguous_df = pd.DataFrame(list(ambiguous_examples), columns=["row_idx", "col_idx"])
ambiguous_df.to_csv("/home/mbrannon/tmp/ambiguous_variants.tsv", sep="\t", index=False)
print(f"Saved {len(ambiguous_df)} ambiguous examples to ambiguous_variants.tsv")

# ------------------------------
# Filter AnnData to remove ambiguous variants
# ------------------------------
rows_to_remove, cols_to_remove = zip(*ambiguous_examples) if ambiguous_examples else ([], [])
train_var_adata_filtered = train_var_adata.copy()
if cols_to_remove:
    train_var_adata_filtered = train_var_adata_filtered[:, [i for i in range(train_var_adata_filtered.n_vars) if i not in cols_to_remove]]

train_var_adata_filtered.write("/home/mbrannon/tmp/train_var_adata_filtered.h5ad")
print("Saved filtered AnnData excluding ambiguous variants.")
