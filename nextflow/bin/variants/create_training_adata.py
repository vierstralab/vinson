import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from genome_tools.data.anndata import read_zarr_backed

LAYER_COLUMNS = ["ref_counts", "total_counts", "BAD", "logit_es"]


def build_variant_training_adata(variants, epochs, reference_anndata, val_chroms=("chr9", "chr21")):
    """
    Same classified variant set is replicated under every epoch's layer suffix: only
    the jitter offsets differ per epoch (added later, by jitter_offsets), not the
    underlying variant calls. The matrices are small - duplicating them is cheap.
    """
    variants["variant_id"] = (
        variants["#chr"] + ":" + variants["end"].astype(str) + ":" + variants["ref"] + ":" + variants["alt"]
    )
    variants['total_counts'] = variants.eval('ref_counts + alt_counts')
    passed = (variants["total_counts"] > 20) & (variants["ref_counts"] > 0) & (variants["alt_counts"] > 0)
    is_positive = passed & (variants["FDR_sample"] < 0.1) & (variants["logit_es"].abs() < 6)
    is_negative = passed & (variants["FDR_sample"] > 0.5)
    variants["class"] = np.where(is_positive, 1, -1)
    variants = variants[is_positive | is_negative].copy()



    reference_anndata.obs_names = reference_anndata.obs_names.astype(str)
    ref_slice = reference_anndata[reference_anndata.obs_names.isin(variants["sample_id"])].copy()

    var = variants.drop_duplicates("variant_id").set_index("variant_id")[["#chr", "end", "ref", "alt"]]
    # masterlist-shaped columns so this anndata can be fed into nf-index's
    # extractDataWithOffsets exactly like a DHS reference: each variant is a 1bp
    # "DHS" with no distinct summit, so summit = start.
    var["start"] = var["end"] - 1
    var["dhs_summit"] = var["start"]
    var["dhs_id"] = var.index
    adata = ad.AnnData(X=None, obs=ref_slice.obs, var=var, obsm=dict(ref_slice.obsm))

    obs_pos = pd.Series(range(adata.n_obs), index=adata.obs_names)
    var_pos = pd.Series(range(adata.n_vars), index=adata.var_names)
    row_idx = variants["sample_id"].map(obs_pos).to_numpy()
    col_idx = variants["variant_id"].map(var_pos).to_numpy()

    for epoch in epochs:
        def add_layer(name, vals):
            adata.layers[f"{name}.{epoch}"] = sp.coo_matrix((vals, (row_idx, col_idx)), shape=adata.shape).tocsr()

        for col in LAYER_COLUMNS:
            add_layer(col, variants[col].to_numpy(dtype=np.float32))

    indiv_id = variants.drop_duplicates("sample_id").set_index("sample_id")["indiv_id"]
    adata.obs["genotype_id"] = adata.obs_names.map(indiv_id)

    # the datamodule only ever requests sample_split == "train"; variants are held out
    # for validation, not samples, so split_data on the var axis is what actually matters.
    adata.obsm["split_data"] = np.full(adata.n_obs, "train", dtype="U5")
    is_val = adata.var["#chr"].isin(val_chroms).to_numpy()
    adata.varm["split_data"] = np.where(is_val, "val", "train").astype("U5")

    adata.uns["epoch_names"] = list(epochs)

    # tested (sample, variant) pairs, shared by every epoch, in the same var x obs
    # on-disk convention as nf-index's matrices, so the existing jitter_offsets
    # process can be reused unmodified (once per epoch, with a different seed).
    class_matrix = sp.coo_matrix(
        (variants["class"].to_numpy(dtype=np.float32), (row_idx, col_idx)), shape=adata.shape
    ).tocsr().T

    return adata, class_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a variant training AnnData, same variants but different pre-jitter offsets per epoch).")
    parser.add_argument("variants_parquet", help="Non-aggregated variant table (parquet)")
    parser.add_argument("reference_anndata", help="Reference AnnData zarr (for obs + motif_embeddings)")
    parser.add_argument("output", help="Output AnnData zarr path")
    parser.add_argument("class_matrix_output", help="Output path for the npz matrix, for jitter_offsets")
    parser.add_argument("sample_names_output", help="Output path for the sample names matching the class matrix columns")
    parser.add_argument("--epoch", nargs="+", required=True, help="Epoch names")
    parser.add_argument("--val_chroms", default="chr9,chr21", help="Comma-separated chromosomes held out for validation")
    args = parser.parse_args()

    variants = pd.read_parquet(args.variants_parquet)
    reference_anndata = read_zarr_backed(args.reference_anndata)

    adata, class_matrix = build_variant_training_adata(
        variants, args.epoch, reference_anndata,
        val_chroms=args.val_chroms.split(','),
    )
    adata.write_zarr(args.output)
    sp.save_npz(args.class_matrix_output, class_matrix, compressed=True)
    np.savetxt(args.sample_names_output, adata.obs_names, fmt='%s')
