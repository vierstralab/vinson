import sys

from genome_tools.data.anndata import (
    read_zarr_backed,
    add_matrices_to_anndata,
    get_matrices_mapping_by_types,
)


def main(adata, matrices):
    matrices_types = ['density', 'mean_bg_agg_cutcounts'] + list(adata.layers.keys())
    matrices_mapping = get_matrices_mapping_by_types(matrices, matrices_types)

    add_matrices_to_anndata(adata, matrices_mapping)

    return adata


if __name__ == '__main__':
    adata_obj = read_zarr_backed(sys.argv[1])
    output = sys.argv[2]
    matrices = sys.argv[3:]

    main(adata_obj, matrices).write_zarr(output)
