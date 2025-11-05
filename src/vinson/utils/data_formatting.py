import numpy as np
import anndata as ad
import h5py
import pandas as pd


def extract_data_from_h5(h5_file, ref_adata: ad.AnnData):
    data_keys = {
        'dhs_id': np.str_,
        'read_depth': np.float32,
        'sample_id': np.str_,
        'chrom': np.str_,
        'summit': np.int32,
        'background': np.float32,
        'class': np.int8,
        'density': np.float32,
    }
    optional_keys = {
        'indiv_id': np.str_,
        'dhs_weight': np.float32,
    }
    with h5py.File(h5_file, 'r') as f:
        data = {}
        for key, dtype in data_keys.items():
            p = f[key][()]
            data[key] = np.ascontiguousarray(p.astype(dtype))
        for key, dtype in optional_keys.items():
            if key in f:
                data[key] = np.ascontiguousarray(f[key][()].astype(dtype))

    return data, ref_adata.obsm['motif_embeddings']


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
