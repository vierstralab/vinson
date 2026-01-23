import numpy as np
import anndata as ad
import h5py
import pandas as pd
import dask.array as da


class VinsonData:
    """
    Class for handling Vinson data formatting and conversion.

    Parameters
    ----------
    data : dict
        Dictionary containing sample metadata. {
            'chrom': np.array, 'summit': np.array, etc.
        }
    encodings : dict
        Dictionary containing encodings for categorical variables. {
            'chrom': np.array, ...
        }
    embeddings_df : pd.DataFrame
        DataFrame of cell-type/state embeddings indexed by sample ID.
    """
    def __init__(self, data: dict, encodings: dict, embeddings_df: pd.DataFrame=None, is_variant=False):
        self.data = data
        self.encodings = encodings
        self.embeddings_df = embeddings_df
        self.is_variant = is_variant

        self.length = len(self.data['chrom'])
        for key, value in self.data.items():
            assert len(value) == self.length, f"All data arrays must have the same length. Key {key} has length {len(value)}, expected {self.length}."
    
    def __repr__(self):
        return f"VinsonData with keys: {list(self.data.keys())}. Encoded columns: {list(self.encodings.keys())}."
    
    def decode(self, key: str) -> np.ndarray:
        return self._decode(key, self.data[key])
    
    def _decode(self, key: str, indices: np.ndarray) -> np.ndarray:
        """
        Decode encoded values for a given key.
        
        Parameters:
            key (str): Key of the data dictionary to decode.
            indices (np.ndarray): Encoded indices to decode.
        """
        if key not in self.encodings:
            raise ValueError(f"Key {key} is not encoded.")
        return self.encodings[key][indices]

    def to_df(self) -> pd.DataFrame:
        """Convert data dictionary to pandas DataFrame."""
        data_df = pd.DataFrame(self.data)
        for key, enc in self.encodings.items():
            data_df[key] = pd.Categorical.from_codes(data_df[key], enc)
        return data_df
    
    def __len__(self):
        """Return number of samples in the dataset."""
        return self.length
    
    def keys(self):
        """Return keys of the data dictionary."""
        return self.data.keys()
    
    def __contains__(self, key):
        """Check if key is in the data dictionary."""
        return key in self.data.keys()

    def __getitem__(self, i) -> dict:
        """Get data dict for a given index."""
        return_dict = {}
        for key, value in self.data.items():
            return_dict[key] = value[i]
            if key in self.encodings:
                return_dict[key] = self._decode(key, return_dict[key])
                
        return return_dict

    def write_h5(self, h5_file: str):
        """Convert data dictionary to H5 file."""
        strings_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(h5_file, 'w') as f:
            for key, value in self.data.items():
                if key in self.encodings:
                    value = np.astype(self.decode(key), strings_dtype)
                f.create_dataset(key, data=value, compression="gzip")

    @classmethod
    def from_raw(cls, raw_data: dict, embeddings_df: pd.DataFrame=None, is_variant=False):
        data, encodings = sanitize_data(raw_data, is_variant=is_variant)
        return cls(data, encodings, embeddings_df, is_variant=is_variant)


def sanitize_data(data: dict, encodings: dict = None, is_variant=False) -> tuple:
    """Ensure that all data arrays are contiguous and of the correct dtype."""
    if encodings is None:
        encodings = {}
    if is_variant:
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
    else:
        data_keys = {
            "read_depth": np.float32,
            "sample_id": np.str_,
            "dhs_id": np.str_,
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
    keys = {
        **data_keys,
        **{x: y for x, y in optional_keys.items() if x in data},
    }
    for key, dtype in keys.items():
        if dtype == np.str_:
            if key in encodings:
                data[key] = np.asarray(data[key], dtype=np.int32)
                encodings[key] = np.asarray(encodings[key], dtype=np.str_)
                mask = pd.isna(encodings[key]) | np.isin(encodings[key], ['None', 'nan'])
                encodings[key][mask] = ''
            else:
                enc, inverse = np.unique(data[key], return_inverse=True)
                data[key] = np.asarray(inverse, dtype=np.int32)
                encodings[key] = np.asarray(enc, dtype=np.str_)
        else:
            data[key] = np.asarray(data[key], dtype=dtype)
        if not data[key].flags["C_CONTIGUOUS"]:
            data[key] = np.ascontiguousarray(data[key])

    if 'background' in data:
        data['background'] = np.nan_to_num(data['background'], copy=False)
    return data, encodings


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


def extract_data_from_train_anndata(train_adata: ad.AnnData, suffix: str) -> VinsonData:
    """
    Convert train AnnData object to H5 format and extract embeddings.
    Args:
        adata (ad.AnnData): AnnData object containing DHS data.
        suffix (str): Suffix of the epoch/layer to extract. Gets added to layer names as `{layer}.{suffix}`. E.g. epoch_1, epoch_2, etc.
    
    Returns:
        data (dict): Dictionary containing extracted data arrays.
        embeddings_df (pd.DataFrame): DataFrame containing motif embeddings.
    """
    
    data = {"class": None, "density": None, "mean_bg_agg_cutcounts": None}
    update_layers_dict(data, train_adata, suffix)
    row_idx, col_idx = get_examples_indices_from_layer(data["class"])
    encodings = {}

    encoded = {}
    encoding_sources = {
        "sample_id": train_adata.obs_names,
        "dhs_id": train_adata.var_names,
        "chrom": train_adata.var["#chr"],
    }

    if "indiv_id" in train_adata.obsm:
        encoding_sources["indiv_id"] = train_adata.obsm["indiv_id"]

    for key, arr in encoding_sources.items():
        enc, inv = np.unique(arr, return_inverse=True)
        encodings[key] = enc
        encoded[key] = inv

    data = {
        'read_depth': train_adata.obs['nuclear_reads'].values[row_idx],
        'sample_id': encoded["sample_id"][row_idx],
        'dhs_id': encoded["dhs_id"][col_idx],
        'chrom': encoded["chrom"][col_idx],
        'summit': train_adata.var['dhs_summit'].values[col_idx],
        'background': data['mean_bg_agg_cutcounts'].data,
        'class': data['class'].data,
        'density': data['density'].data,
    }
    if 'indiv_id' in encoded:
        data['indiv_id'] = encoded["indiv_id"][row_idx]

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
    
    layers = {"ref_counts": None, "total_counts": None, "BAD": None, "logit_es": None}
    update_layers_dict(layers, train_adata, suffix)
    row_idx, col_idx = get_examples_indices_from_layer(layers["ref_counts"])

    data = {
        'chrom': train_adata.var['#chr'].values[col_idx],
        'pos': train_adata.var['end'].values[col_idx],
        'ref': train_adata.var['ref'].values[col_idx],
        'alt': train_adata.var['alt'].values[col_idx],
        'sample_id': train_adata.obs_names[row_idx],
        'ref_counts': layers['ref_counts'].data,
        'total_counts': layers['total_counts'].data,
        'BAD': layers['BAD'].data,
        'logit_es': layers['logit_es'].data,
    }

    if 'indiv_id' in train_adata.obsm:
        data['indiv_id'] = train_adata.obsm['indiv_id'][row_idx]

    return VinsonData.from_raw(
        data,
        train_adata.obsm['motif_embeddings'],
        is_variant=True
    )


def extract_data_from_backed_anndata(backed_anndata, dhs_ids=None, sample_ids=None, use_sample_peaks=False,
                                     extra_layers=()) -> VinsonData:
    """
    This function can also be used to extract data into training anndata object.
    """
    # FIXME: Not optimized yet
    adata_slice = slice_adata(backed_anndata, dhs_ids, sample_ids) # sample x dhs

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

    embeddings_df = adata_slice.obsm['motif_embeddings']
    return VinsonData.from_raw(
        data,
        embeddings_df,
        is_variant=False
    )


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


def get_examples_indices_from_layer(layer_coo):
    row_idx, col_idx = layer_coo.row, layer_coo.col
    return row_idx, col_idx

def update_layers_dict(layers: dict, train_adata: ad.AnnData, suffix: str):
    assert len(layers) > 0, "Must provide at least one layer to extract"
    for layer_name in layers:
        epoch_layer_name = f"{layer_name}.{suffix}"
        layers[layer_name] = train_adata.layers[epoch_layer_name].tocoo()


def get_number_of_train_examples(anndata_file):
    adata = ad.read_h5ad(anndata_file)
    if 'n_training_examples' in adata.uns:
        n_examples = adata.uns['n_training_examples']
    else:
        n_examples = 0
        for epoch in adata.uns['epoch_names']:
            layer_name = f"class.{epoch}"
            if layer_name not in adata.layers:
                raise ValueError(f"Layer {layer_name} not found in AnnData layers. Cannot determine number of training examples.")
            layer = adata.layers[layer_name]
            n_examples += layer.getnnz()

    return n_examples

def long_to_wide_multitask_vinsondata(anndata_obj: ad):

    vd_long = extract_data_from_backed_anndata(anndata_obj)
    d = vd_long.data
    
    # decode string-coded arrays if encoded
    dhs_all = vd_long.decode("dhs_id") if "dhs_id" in vd_long.encodings else d["dhs_id"]
    samp_all = vd_long.decode("sample_id") if "sample_id" in vd_long.encodings else d["sample_id"]
    chrom_all = vd_long.decode("chrom") if "chrom" in vd_long.encodings else d["chrom"]
    
    dhs_ids_unique = np.unique(dhs_all)
    n_dhs = len(dhs_ids_unique)

    N = len(dhs_all)
    if N % n_dhs != 0:
        raise ValueError(f"N={N} not divisible by n_dhs={n_dhs}")
    n_samples = N // n_dhs

    # verify sample-major blocks
    if len(np.unique(samp_all[:n_dhs])) != 1:
        raise ValueError("Ordering is not sample-major blocks; cannot reshape safely.")

    # sample order
    sample_ids_1d = samp_all[::n_dhs]
    read_depth_1d = d["read_depth"][::n_dhs]

    # reshape numeric arrays -> wide (n_dhs, n_samples)
    density = d["density"].reshape(n_samples, n_dhs).T
    background = d["background"].reshape(n_samples, n_dhs).T
    cls = d["class"].reshape(n_samples, n_dhs).T

    # per-DHS metadata from first block (already decoded for chrom/dhs_id)
    chrom = chrom_all[:n_dhs]
    summit = d["summit"][:n_dhs]
    dhs_id = dhs_all[:n_dhs]

    # broadcast sample info to satisfy sanitize_data requirements
    sample_id = np.tile(sample_ids_1d.reshape(1, -1), (n_dhs, 1))
    read_depth = np.tile(read_depth_1d.reshape(1, -1), (n_dhs, 1))

    raw_wide = {
        "chrom": chrom,                 # (n_dhs,) strings like "chr1"
        "summit": summit,               # (n_dhs,)
        "dhs_id": dhs_id,               # (n_dhs,) strings
        "density": density,             # (n_dhs, n_samples)
        "background": background,       # (n_dhs, n_samples)
        "class": cls,                   # (n_dhs, n_samples)
        "sample_id": sample_id,         # (n_dhs, n_samples) strings
        "read_depth": read_depth,       # (n_dhs, n_samples) float
    }

    vd_wide = VinsonData.from_raw(raw_wide, vd_long.embeddings_df, is_variant=vd_long.is_variant)
    return vd_wide, sample_ids_1d, read_depth_1d