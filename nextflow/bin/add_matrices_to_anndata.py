import argparse
import os
import re

import dask.array as da
import numpy as np
import scipy.sparse as sp

from genome_tools.data.anndata import read_zarr_backed, add_matrices_to_anndata

# 'matrix.<tag>.<matrix_type>.npy|npz' -> layer key '<matrix_type>.<tag>'.
# <tag> is usually an epoch id, but the parsing itself doesn't care what it represents.
MATRIX_NAME_RE = re.compile(r'^matrix\.(?P<tag>.+)\.(?P<matrix_type>[^.]+)\.(?:npy|npz)$')


def get_matrices_mapping_by_tag(matrices, matrices_types):
    mapping = {}
    tags = []
    for matrix in matrices:
        match = MATRIX_NAME_RE.match(os.path.basename(matrix))
        if match is None or match['matrix_type'] not in matrices_types:
            raise ValueError(
                f"Matrix {matrix} does not match the expected "
                f"'matrix.<tag>.<matrix_type>.npy|npz' naming. Expected matrix_type in {matrices_types}"
            )
        mapping[f"{match['matrix_type']}.{match['tag']}"] = matrix
        if match['tag'] not in tags:
            tags.append(match['tag'])
    return mapping, tags


def align_to_class_layer(layer, class_layer):
    """
    Store `layer` with exactly the sparsity pattern of `class_layer` (explicit zeros included),
    so training code can match layers element-wise via `.data`.
    """
    coo = class_layer.tocoo()
    if sp.issparse(layer):
        values = np.asarray(layer.tocsr()[coo.row, coo.col]).ravel()
    else:
        values = np.asarray(layer[coo.row, coo.col]).ravel()
    return sp.csr_matrix(
        (values.astype(layer.dtype), class_layer.indices.copy(), class_layer.indptr.copy()),
        shape=class_layer.shape,
    )


def main(adata, matrices, val_chroms=("chr9", "chr21")):
    matrices_types = ['density', 'mean_bg_agg_cutcounts', 'class', 'offsets'] + list(adata.layers.keys())
    matrices_mapping, epochs = get_matrices_mapping_by_tag(matrices, matrices_types)

    # remove dense layers
    adata.X = None
    for layer_name in list(adata.layers.keys()):
        if not sp.issparse(adata.layers[layer_name]):
            print(f"Dropping dense layer {layer_name}")
            del adata.layers[layer_name]

    add_matrices_to_anndata(adata, matrices_mapping)

    # downstream training code (vinson.utils.data_formatting) calls .tocoo() on these
    # layers unconditionally, so they must be sparse regardless of the generic
    # sufficiently-sparse heuristic (can leave them dense on smaller datasets).
    for layer_name in matrices_mapping:
        if layer_name.startswith('class.'):
            class_layer = sp.csr_matrix(adata.layers[layer_name])
            class_layer.sort_indices()
            adata.layers[layer_name] = class_layer

    # and they're read via .data, so every other layer of an epoch must store exactly the
    # class layer's entries; a plain dense->sparse conversion drops sampled examples whose
    # value is 0 (e.g. density of a negative with no reads, or a jitter offset of 0).
    for layer_name in matrices_mapping:
        matrix_type, tag = layer_name.split('.', 1)
        class_key = f"class.{tag}"
        if matrix_type != 'class' and class_key in adata.layers:
            adata.layers[layer_name] = align_to_class_layer(adata.layers[layer_name], adata.layers[class_key])
        elif not sp.issparse(adata.layers[layer_name]):
            adata.layers[layer_name] = sp.csr_matrix(adata.layers[layer_name])

    # all samples are used for training; DHSs on val_chroms are held out for validation
    adata.obsm['split_data'] = np.full(adata.n_obs, 'train')
    adata.varm['split_data'] = np.where(adata.var['#chr'].isin(val_chroms).to_numpy(), 'val', 'train')

    existing_epochs = list(adata.uns.get('epoch_names', []))
    adata.uns['epoch_names'] = existing_epochs + [e for e in epochs if e not in existing_epochs]

    # leave n_training_examples untouched when there's no class layer.
    class_epochs = [e for e in adata.uns['epoch_names'] if f"class.{e}" in adata.layers]
    if class_epochs:
        adata.uns['n_training_examples'] = sum(adata.layers[f"class.{e}"].getnnz() for e in class_epochs)

    return adata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add per-epoch matrices to an AnnData and save it as a training h5ad")
    parser.add_argument("anndata", help="Source AnnData zarr")
    parser.add_argument("output", help="Output h5ad path")
    parser.add_argument("matrices", nargs="+", help="'matrix.<tag>.<matrix_type>.npy|npz' files")
    parser.add_argument("--val_chroms", default="chr9,chr21", help="Comma-separated chromosomes held out for validation")
    args = parser.parse_args()

    adata_obj = read_zarr_backed(args.anndata)
    output = args.output

    adata_obj = main(adata_obj, args.matrices, val_chroms=args.val_chroms.split(','))

    # read_zarr_backed loads varm lazily as dask arrays regardless of their dtype;
    # h5ad can't write those (or object-dtype string arrays), so materialize first.
    for key, value in list(adata_obj.varm.items()):
        if isinstance(value, da.Array):
            value = value.compute()
        if getattr(value, 'dtype', None) == object:
            value = value.astype(str)
        adata_obj.varm[key] = value

    adata_obj.write_h5ad(output)
