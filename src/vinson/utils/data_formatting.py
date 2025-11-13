import numpy as np
import anndata as ad
import h5py
import pandas as pd


def extract_var_data_from_h5(h5_file, ref_adata: ad.AnnData):
    """Extract variant-level data (with ref/alt info) from an H5 file."""
    data_keys = {
        "chrom": np.str_,
        "pos": np.int32,
        "ref": np.str_,
        "alt": np.str_,
        "ref_counts": np.float32,
        "total_counts": np.float32,
        "BAD": np.float32,
        "sample_id": np.str_,
        "logit_es": np.float32,
    }
    optional_keys = {
        "indiv_id": np.str_,
        "dhs_weight": np.float32,
    }

    data = {}
    with h5py.File(h5_file, "r") as f:
        for key, dtype in data_keys.items():
            if key not in f:
                raise KeyError(f"Missing expected key '{key}' in {h5_file}")
            data[key] = np.ascontiguousarray(f[key][()].astype(dtype))

        for key, dtype in optional_keys.items():
            if key in f:
                data[key] = np.ascontiguousarray(f[key][()].astype(dtype))

        if "indiv_id" not in data:
            indiv_map = pd.read_table(
                "/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/metadata.clustered.tsv"
            ).set_index("sample_id")["indiv_id"]
            data["indiv_id"] = pd.Series(data["sample_id"]).map(indiv_map).values

    return data, ref_adata.obsm["motif_embeddings"]


def extract_data_from_h5(h5_file, ref_adata: ad.AnnData):
    """Extract DHS-level (non-variant) data from an H5 file."""
    data_keys = {
        "dhs_id": np.str_,
        "read_depth": np.float32,
        "sample_id": np.str_,
        "chrom": np.str_,
        "summit": np.int32,
        "background": np.float32,
        "class": np.int8,
        "density": np.float32,
    }
    optional_keys = {
        "indiv_id": np.str_,
        "dhs_weight": np.float32,
    }

    data = {}
    with h5py.File(h5_file, "r") as f:
        for key, dtype in data_keys.items():
            if key not in f:
                raise KeyError(f"Missing expected key '{key}' in {h5_file}")
            data[key] = np.ascontiguousarray(f[key][()].astype(dtype))

        for key, dtype in optional_keys.items():
            if key in f:
                data[key] = np.ascontiguousarray(f[key][()].astype(dtype))

    return data, ref_adata.obsm["motif_embeddings"]


#add functionality for variants
def extract_data_from_anndata(adata: ad.AnnData, suffix: str):
    """
    Convert AnnData object to H5 format and extract embeddings.
    Args:
        adata (ad.AnnData): AnnData object containing DHS data.
        suffix (str): Suffix of the epoch/layer to extract. Gets added to layer names as `{layer}.{suffix}`. E.g. epoch_1, epoch_2, etc.
    
    Returns:
        data (dict): Dictionary containing extracted data arrays.
        embeddings_df (pd.DataFrame): DataFrame containing motif embeddings.
    """
    
    layers = {"class": None, "density": None, "mean_bg_agg_cutcounts": None}
    for layer_name in layers:
        epoch_layer_name = f"{layer_name}.{suffix}"
        layers[layer_name] = adata.layers[epoch_layer_name].tocoo()
        # for epoch_name in self.epoch_names:
        #    del full_adata.layers[f"{layer_name}.{epoch_name}"]

    class_coo = layers["class"]
    row_idx, col_idx = class_coo.row, class_coo.col

    data = {
        'read_depth': adata.obs['nuclear_reads'].values[row_idx],
        'sample_id': adata.obs_names[row_idx],
        'chrom': adata.var['#chr'].values[col_idx],
        'summit': adata.var['dhs_summit'].values[col_idx],
        'background': layers['mean_bg_agg_cutcounts'].data,
        'class': layers['class'].data,
        'density': layers['density'].data,
    }

    if 'indiv_id' in adata.obsm:
        # maybe come up with something more elegant
        indiv_ids = np.array(
            [
                x if x != "None" else None
                for x in adata.obsm['indiv_id']
            ]
        )
        data['indiv_id'] = indiv_ids[row_idx]

    if 'dhs_weight' in adata.varm:
        data['dhs_weight'] = adata.varm['dhs_weight'][col_idx]

    data = {k: np.ascontiguousarray(v) for k, v in data.items()}

    embeddings_df = adata.obsm['motif_embeddings']
    return data, embeddings_df

def extract_var_data_from_anndata(adata: ad.AnnData, suffix: str):
    """
    Convert AnnData object to H5 format and extract embeddings.
    Args:
        adata (ad.AnnData): AnnData object containing DHS data.
        suffix (str): Suffix of the epoch/layer to extract. Gets added to layer names as `{layer}.{suffix}`. E.g. epoch_1, epoch_2, etc.
    
    Returns:
        data (dict): Dictionary containing extracted data arrays.
        embeddings_df (pd.DataFrame): DataFrame containing motif embeddings.
    """
    
    layers = {"ref_counts": None, "total_counts": None, "BAD": None, "logit_es":None}
    for layer_name in layers:
        epoch_layer_name = f"{layer_name}.{suffix}"
        layers[layer_name] = adata.layers[epoch_layer_name].tocoo()

    class_coo = layers["total_counts"]
    row_idx, col_idx = class_coo.row, class_coo.col

    data = {
        'chrom': adata.var['#chr'].values[col_idx],
        'pos': adata.var['pos'].values[col_idx],
        'ref': adata.var['ref'].values[col_idx],
        'alt': adata.var['alt'].values[col_idx],
        'sample_id': adata.obs_names[row_idx],
        'ref_counts': layers['ref_counts'].data,
        'total_counts': layers['total_counts'].data,
        'BAD': layers['BAD'].data,
        'logit_es': layers['logit_es'].data,
    }

    if 'indiv_id' in adata.obsm:
        # maybe come up with something more elegant
        indiv_ids = np.array(
            [
                x if x != "None" else None
                for x in adata.obsm['indiv_id']
            ]
        )
        data['indiv_id'] = indiv_ids[row_idx]

    data = {k: np.ascontiguousarray(v) for k, v in data.items()}

    embeddings_df = adata.obsm['motif_embeddings']
    return data, embeddings_df
