

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
        tuple val(meta), val(epoch), path(name)
    
    script:
    name = "${epoch}.sampled_examples_mat.npz"
    """
    python3 $moduleDir/bin/sample_examples.py \
        ${meta.zarr_anndata} \
        ${params.sampling_dist} \
        ${name} \
        --seed ${meta.seed}
    """
}


process jitter_offsets {
    conda "${params.conda}"
    tag "${epoch}"

    input:
        tuple val(meta), val(epoch), path(examples)
    
    output:
        tuple val(meta), val(epoch), path(name)
    
    script:
    name = "${epoch}.sampled_offsets.npz"
    """
    python3 $moduleDir/bin/sample_examples.py \
        ${examples} \
        ${name} \
        --seed ${meta.seed} \
        --jitter ${params.jitter}
    """
}



// process save_epochs_data {
//     conda "${params.conda}"

//     input:
//         tuple val(meta), val(epoch)

//     output:
//         path name

//     script:

// }

// process assemble_train_anndata {

// }


workflow {
    Channel.fromPath(params.epochs_meta)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, it.epoch))
        | sample_epoch
        | jitter_offsets
        | map(it -> tuple(it[1], it[3], it[0].zarr_anndata)) //epoch offsets, anndata
        | extractDataWithOffsets

}
