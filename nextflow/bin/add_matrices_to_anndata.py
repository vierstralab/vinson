import os
import re
import sys

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
    matrices_types = ['density', 'mean_bg_agg_cutcounts', 'class'] + list(adata.layers.keys())
    matrices_mapping, epochs = get_matrices_mapping_by_tag(matrices, matrices_types)

    add_matrices_to_anndata(adata, matrices_mapping)

    existing_epochs = list(adata.uns.get('epoch_names', []))
    adata.uns['epoch_names'] = existing_epochs + [e for e in epochs if e not in existing_epochs]

    adata.uns['n_training_examples'] = sum(
        adata.layers[f"class.{epoch}"].getnnz()
        for epoch in adata.uns['epoch_names']
        if f"class.{epoch}" in adata.layers
    )

    return adata


if __name__ == '__main__':
    adata_obj = read_zarr_backed(sys.argv[1])
    output = sys.argv[2]
    matrices = sys.argv[3:]

    main(adata_obj, matrices).write_h5ad(output)
