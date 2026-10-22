

include { extractDataWithOffsets } from "./pipelines/nf-index/extract_data_with_offsets" addParams(conda: params.conda_index)


params.sampling_dist = 0.38
params.jitter = 40


process sample_epoch {
    conda "${params.conda}"
    tag "${epoch}"
    cpus 10

    input:
        tuple val(meta), val(epoch)
    
    output:
        tuple val(meta), val(epoch), path(sampled_examples), path(sample_names)
    
    script:
    sampled_examples = "${epoch}.sampled_examples_mat.npz"
    sample_names = "${epoch}.sample_names.txt"
    """
    python3 $moduleDir/bin/sample_examples.py \
        ${meta.zarr_anndata} \
        ${params.sampling_dist} \
        ${sampled_examples} \
        ${sample_names} \
        --seed ${meta.seed}
    """
}


process jitter_offsets {
    conda "${params.conda}"
    tag "${epoch}"

    input:
        tuple val(meta), val(epoch), path(examples), path(sample_names)
    
    output:
        tuple val(meta), val(epoch), path(name)
    
    script:
    name = "${epoch}.sampled_offsets.npz"
    """
    python3 $moduleDir/bin/jitter_offsets.py \
        ${examples} \
        ${sample_names} \
        ${name} \
        --seed ${meta.seed} \
        --jitter ${params.jitter}
    """
}

process assemble_train_anndata {
    conda "${params.conda}"
    publishDir "${params.outdir}/train_data"

    input:
        tuple val(zarr_anndata), val(prefix), path(matrices)

    output:
        tuple val(prefix), path(name)

    script:
    name = "${prefix}.anndata.zarr"
    """
    python3 $moduleDir/bin/add_matrices_to_anndata.py \
        ${zarr_anndata} \
        ${name} \
        ${matrices} \
    """
}


workflow {
    input_data = Channel.fromPath(params.epochs_meta)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, it.epoch))

    input_data
        | sample_epoch
        | jitter_offsets
        | map(it -> tuple(it[1], it[2], it[0].zarr_anndata)) //epoch offsets, anndata
        | extractDataWithOffsets
        | combine(input_data)
        | map(it -> tuple(it[2].zarr_anndata, it[2].prefix, it[1])) // zarr_anndata, prefix, matrix
        | unique()
        | groupTuple(by: [0, 1])
        | assemble_train_anndata
}
