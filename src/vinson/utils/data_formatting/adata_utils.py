import dask.array as da
import anndata as ad


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
    full_adata = ad.read_h5ad(anndata_file)
    adata = full_adata[
        full_adata.obsm["split_data"] == "train",
        full_adata.varm["split_data"] == "train",
    ]
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