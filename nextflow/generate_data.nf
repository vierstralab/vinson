
process generate_cell_selective_data {

    conda "${params.conda}"
    publishDir "${params.outdir}/"

    output:
        path "*.${suffix}"

    script:
    suffix = "cell_selective_data.h5"
    """
    python3 $moduleDir/bin/generate_cell_selective_data.py \
        --adata ${params.zarr_anndata} \
        --suffix ${suffix}
    """
}



