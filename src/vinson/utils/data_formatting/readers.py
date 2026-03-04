import numpy as np
import anndata as ad
import h5py
import pandas as pd

from .adata_utils import slice_adata, compute_if_dask, get_examples_indices_from_layer, update_layers_dict
from .encoding import sanitize_data, encode_inplace

from .container import VinsonData


def extract_data_from_h5(h5_file, ref_adata: ad.AnnData, is_variant=False) -> VinsonData:
    with h5py.File(h5_file, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][()]

    return VinsonData.from_raw(
        data,
        ref_adata.obsm['motif_embeddings'],
        is_variant=is_variant
    )


def extract_data_from_train_anndata(train_adata: ad.AnnData, suffix: str, pre_jitter=False) -> VinsonData:
    """
    Convert train AnnData object to H5 format and extract embeddings.
    Args:
        adata (ad.AnnData): AnnData object containing DHS data.
        suffix (str): Suffix of the epoch/layer to extract. Gets added to layer names as `{layer}.{suffix}`. E.g. epoch_1, epoch_2, etc.
        pre_jitter (bool): Whether to apply pre-jitter to the data. 
    
    Returns:
        data (dict): Dictionary containing extracted data arrays.
        embeddings_df (pd.DataFrame): DataFrame containing motif embeddings.
    """
    data = {"class": None, "density": None, "mean_bg_agg_cutcounts": None}
    
    if pre_jitter:
        data["offsets"] = None
    print(data.keys())
    update_layers_dict(data, train_adata, suffix)
    row_idx, col_idx = get_examples_indices_from_layer(data["class"])
    encodings = {}

    encoded_vals = {
        "sample_id": train_adata.obs_names,
        "dhs_id": train_adata.var_names,
        "chrom": train_adata.var["#chr"],
    }

    if "indiv_id" in train_adata.obsm:
        encoded_vals["indiv_id"] = train_adata.obsm["indiv_id"]

    for key in encoded_vals:
        encode_inplace(encoded_vals, encodings, key)


    data = {
        'read_depth': train_adata.obs['nuclear_reads'].values[row_idx],
        'sample_id': encoded_vals["sample_id"][row_idx],
        'dhs_id': encoded_vals["dhs_id"][col_idx],
        'summit': train_adata.var['dhs_summit'].values[col_idx],
        'chrom': encoded_vals["chrom"][col_idx],
        'background': data['mean_bg_agg_cutcounts'].data,
        'class': data['class'].data,
        'density': data['density'].data,
    }

    if pre_jitter:
        data['summit'] += data['offsets'].data

    if 'indiv_id' in encoded_vals:
        data['indiv_id'] = encoded_vals["indiv_id"][row_idx]

    if 'dhs_weight' in train_adata.varm:
        data['dhs_weight'] = train_adata.varm['dhs_weight'][col_idx]
    
    data, encodings = sanitize_data(data, encodings, is_variant=False)

    embeddings_df = train_adata.obsm['motif_embeddings']
    return VinsonData(
        data,
        encodings=encodings,
        embeddings_df=embeddings_df,
        is_variant=False
    )


def extract_variant_data_from_anndata(train_adata: ad.AnnData, suffix: str) -> VinsonData:
    """
    Convert AnnData object to H5 format and extract embeddings.
    Args:
        adata (ad.AnnData): AnnData object containing DHS data.
        suffix (str): Suffix of the epoch/layer to extract. Gets added to layer names as `{layer}.{suffix}`. E.g. epoch_1, epoch_2, etc.
    
    Returns:
        data (dict): Dictionary containing extracted data arrays.
        embeddings_df (pd.DataFrame): DataFrame containing motif embeddings.
    """
    
    data = {"ref_counts": None, "total_counts": None, "BAD": None, "logit_es": None}
    update_layers_dict(data, train_adata, suffix)
    row_idx, col_idx = get_examples_indices_from_layer(data["ref_counts"])
    
    encodings = {}

    encoding_sources = {
        "sample_id": train_adata.obs_names,
        "chrom": train_adata.var["#chr"],
        'ref': train_adata.var['ref'],
        'alt': train_adata.var['alt']
    }
    if 'indiv_id' in train_adata.obsm:
        encoding_sources['indiv_id'] = train_adata.obsm['indiv_id']

    for key in encoding_sources:
        encode_inplace(encoding_sources, encodings, key)

    data = {
        'chrom': encoding_sources['chrom'][col_idx],
        'pos': data['pos'][col_idx],
        'ref': encoding_sources['ref'][col_idx],
        'alt': encoding_sources['alt'][col_idx],
        'sample_id': encoding_sources["sample_id"][row_idx],
        'ref_counts': data['ref_counts'].data,
        'total_counts': data['total_counts'].data,
        'BAD': data['BAD'].data,
        'logit_es': data['logit_es'].data,
    }
    if 'indiv_id' in encoding_sources:
        data['indiv_id'] = encoding_sources["indiv_id"][row_idx]

    data, encodings = sanitize_data(data, encodings, is_variant=True)
    embeddings_df = train_adata.obsm['motif_embeddings']

    return VinsonData(
        data,
        encodings=encodings,
        embeddings_df=embeddings_df,
        is_variant=True
    )


def _extract_wide_raw(
    backed_anndata,
    dhs_ids=None,
    sample_ids=None,
    extra_layers=(),
    with_embeddings=True
):
    adata_slice = slice_adata(backed_anndata, dhs_ids, sample_ids)
    
    data = {
        "density": compute_if_dask(adata_slice.layers["density"]).T,
        "background": compute_if_dask(adata_slice.layers["mean_bg_agg_cutcounts"]).T,
        "class": np.where(adata_slice.layers["binary"].toarray().T, 1, -1),
        "read_depth": adata_slice.obs["nuclear_reads"].values,
        "summit": adata_slice.var["dhs_summit"].values,
        "sample_id": adata_slice.obs_names.values,
        "dhs_id": adata_slice.var_names.values,
        "chrom": adata_slice.var["#chr"].values,
    }

    to_encode_keys = ["sample_id", "dhs_id", "chrom"]

    if "indiv_id" in adata_slice.obsm:
        data["indiv_id"] = pd.Series(adata_slice.obsm["indiv_id"], index=adata_slice.obs_names)
        to_encode_keys.append("indiv_id")
    
    encodings = {}
    for key in to_encode_keys:
        encode_inplace(data, encodings, key)

    for layer in extra_layers:
        data[layer] = compute_if_dask(adata_slice.layers[layer]).T

    if with_embeddings:
        return data, encodings, adata_slice.obsm["motif_embeddings"]
    return data, encodings, None


def extract_data_from_backed_anndata(
        backed_anndata: ad.AnnData,
        dhs_ids=None,
        sample_ids=None,
        use_sample_peaks=False,
        extra_layers=()
    ) -> VinsonData:
    """
    This function can also be used to extract data into training anndata object.
    """
    data, encodings, embeddings_df = _extract_wide_raw(
        backed_anndata,
        dhs_ids=dhs_ids,
        sample_ids=sample_ids,
        extra_layers=extra_layers,
        with_embeddings=True
    )

    sample_names = data["sample_id"]
    dhs_names = data["dhs_id"]

    # Create broadcasted grids (C order = sample-major)
    broadcasted_dhs_ids, broadcasted_sample_ids = np.meshgrid(
        dhs_names,
        sample_names,
        indexing="ij"
    )

    data['sample_id'] = broadcasted_sample_ids.flatten()
    data['dhs_id'] = broadcasted_dhs_ids.flatten()

    for layer in ['density', 'background', 'class', *extra_layers]:
        data[layer] = data[layer].flatten()
    

    for key in ['indiv_id', 'read_depth']:
        if key not in data:
            continue
        mapping = pd.Series(data[key], index=sample_names)
        data[key] = pd.Series(data['sample_id']).map(mapping)

    for key in ['chrom', 'summit']:
        mapping = pd.Series(data[key], index=dhs_names)
        data[key] = pd.Series(data['dhs_id']).map(mapping)

    if use_sample_peaks:
        sample_peaks_mask = data['class'] == 1
        for key in data:
            data[key] = data[key][sample_peaks_mask]
    
    data, encodings = sanitize_data(data, encodings, is_variant=False)

    return VinsonData(
        data,
        encodings=encodings,
        embeddings_df=embeddings_df,
        is_variant=False
    )

def extract_data_from_backed_anndata_wide(
    backed_anndata,
    dhs_ids=None,
    sample_ids=None,
    extra_layers=()
) -> VinsonData:
    data, encodings, _ = _extract_wide_raw(
        backed_anndata,
        dhs_ids=dhs_ids,
        sample_ids=sample_ids,
        extra_layers=extra_layers,
        with_embeddings=False
    )

    data["sample_id"] = np.tile(
        data["sample_id"],
        len(data["dhs_id"])
    )

    data, encodings = sanitize_data(data, encodings, is_variant=False)

    return VinsonData(
        data,
        encodings=encodings,
        is_variant=False,
    )
