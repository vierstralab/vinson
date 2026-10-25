import os
import re
import sys

import dask.array as da
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


def main(adata, matrices):
    matrices_types = ['density', 'mean_bg_agg_cutcounts', 'class', 'offsets'] + list(adata.layers.keys())
    matrices_mapping, epochs = get_matrices_mapping_by_tag(matrices, matrices_types)

    add_matrices_to_anndata(adata, matrices_mapping)

    # downstream training code (vinson.utils.data_formatting) calls .tocoo() on these
    # layers unconditionally, so they must be sparse regardless of the generic
    # sufficiently-sparse heuristic (can leave them dense on smaller datasets).
    for layer_name in matrices_mapping:
        if not sp.issparse(adata.layers[layer_name]):
            adata.layers[layer_name] = sp.csr_matrix(adata.layers[layer_name])

    existing_epochs = list(adata.uns.get('epoch_names', []))
    adata.uns['epoch_names'] = existing_epochs + [e for e in epochs if e not in existing_epochs]

    # only DHS/sequence training data has a 'class' layer; variants rely on obsm/varm
    # split_data instead, so leave n_training_examples untouched when there's no class layer.
    class_epochs = [e for e in adata.uns['epoch_names'] if f"class.{e}" in adata.layers]
    if class_epochs:
        adata.uns['n_training_examples'] = sum(adata.layers[f"class.{e}"].getnnz() for e in class_epochs)

    return adata


if __name__ == '__main__':
    adata_obj = read_zarr_backed(sys.argv[1])
    output = sys.argv[2]
    matrices = sys.argv[3:]

    adata_obj = main(adata_obj, matrices)

    # read_zarr_backed loads varm lazily as dask arrays regardless of their dtype;
    # h5ad can't write those (or object-dtype string arrays), so materialize first.
    for key, value in list(adata_obj.varm.items()):
        if isinstance(value, da.Array):
            value = value.compute()
        if getattr(value, 'dtype', None) == object:
            value = value.astype(str)
        adata_obj.varm[key] = value

    adata_obj.write_h5ad(output)
