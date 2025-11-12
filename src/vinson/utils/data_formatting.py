import numpy as np
import anndata as ad
import h5py
import pandas as pd
import dask.array as da


def sanitize_data(data: dict) -> dict:
    """Ensure that all data arrays are contiguous and of the correct dtype."""
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
    for key, dtype in data_keys.items():
        data[key] = np.ascontiguousarray(data[key].astype(dtype))
    for key, dtype in optional_keys.items():
        if key in data:
            data[key] = np.ascontiguousarray(data[key].astype(dtype))
    return data


def data_to_h5(h5_file: str, data: dict):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value, compression="gzip")


def extract_data_from_h5(h5_file, ref_adata: ad.AnnData):
    with h5py.File(h5_file, 'r') as f:
        data = {}
        for key in f.keys():
            data[key] = f[key][()]

        data = sanitize_data(data)

    return data, ref_adata.obsm['motif_embeddings']


def extract_data_from_train_anndata(train_adata: ad.AnnData, suffix: str):
    """
    Convert train AnnData object to H5 format and extract embeddings.
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
        layers[layer_name] = train_adata.layers[epoch_layer_name].tocoo()
        # for epoch_name in self.epoch_names:
        #    del full_adata.layers[f"{layer_name}.{epoch_name}"]

    class_coo = layers["class"]
    row_idx, col_idx = class_coo.row, class_coo.col

    data = {
        'read_depth': train_adata.obs['nuclear_reads'].values[row_idx],
        'sample_id': train_adata.obs_names[row_idx],
        'chrom': train_adata.var['#chr'].values[col_idx],
        'summit': train_adata.var['dhs_summit'].values[col_idx],
        'background': layers['mean_bg_agg_cutcounts'].data,
        'class': layers['class'].data,
        'density': layers['density'].data,
    }

    if 'indiv_id' in train_adata.obsm:
        # maybe come up with something more elegant
        indiv_ids = np.array(
            [
                x if x != "None" else None
                for x in train_adata.obsm['indiv_id']
            ]
        )
        data['indiv_id'] = indiv_ids[row_idx]

    if 'dhs_weight' in train_adata.varm:
        data['dhs_weight'] = train_adata.varm['dhs_weight'][col_idx]

    data = {k: np.ascontiguousarray(v) for k, v in data.items()}

    embeddings_df = train_adata.obsm['motif_embeddings']
    return data, embeddings_df


def slice_adata(adata, dhs_ids, sample_ids) -> ad.AnnData:
    adata_slice = adata[:, :]
    if dhs_ids is not None:
        adata_slice = adata_slice[:, dhs_ids]
    if sample_ids is not None:
        adata_slice = adata_slice[sample_ids, :]
    return adata_slice


def compute_if_dask(array):
    if isinstance(array, da.Array):
        array = array.compute()
    return array


def extract_data_from_backed_anndata(backed_anndata, dhs_ids=None, sample_ids=None, use_sample_peaks=False,
                                     extra_layers=()) -> dict:
    """
    This function can also be used to extract data into training anndata object.
    """
    adata_slice = slice_adata(backed_anndata, dhs_ids, sample_ids) # sample x dhs

    print('compute on density', flush=True)
    sample_names = np.array(adata_slice.obs_names)
    dhs_names = np.array(adata_slice.var_names)

    # Create broadcasted grids (C order = sample-major)
    broadcasted_sample_ids, broadcasted_dhs_ids = np.meshgrid(
        sample_names,
        dhs_names,
        indexing="ij"
    )

    sample_to_indiv_mapping = pd.Series(adata_slice.obsm['indiv_id'], index=adata_slice.obs_names)

    data = {}

    data['density'] = compute_if_dask(adata_slice.layers["density"]).flatten()
    data['background'] = compute_if_dask(adata_slice.layers["mean_bg_agg_cutcounts"]).flatten()
    for layer in extra_layers:
        data[layer] = compute_if_dask(adata_slice.layers[layer]).flatten()
    data['class'] = np.where(adata_slice.layers["binary"].toarray().flatten(), 1, -1)
    data['sample_id'] = broadcasted_sample_ids.flatten()
    data['dhs_id'] = broadcasted_dhs_ids.flatten()
    data['read_depth'] = pd.Series(data['sample_id']).map(adata_slice.obs['nuclear_reads'])
    data['indiv_id'] = pd.Series(data['sample_id']).map(sample_to_indiv_mapping)
    data['chrom'] = pd.Series(data['dhs_id']).map(adata_slice.var['#chr'])
    data['summit'] = pd.Series(data['dhs_id']).map(adata_slice.var['dhs_summit'])

    if use_sample_peaks:
        print('using sample peaks', flush=True)
        sample_peaks_mask = data['class'] == 1
        for key in data:
            data[key] = data[key][sample_peaks_mask]

    data = sanitize_data(data)
    return data