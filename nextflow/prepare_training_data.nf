include { extractDataWithOffsets } from "./pipelines/nf-index/extract_data_with_offsets"


params.sampling_dist = 0.38
params.jitter = 40


process sample_epoch {
    conda "${params.conda}"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(meta), val(epoch)
    
    output:
        tuple val(meta), val(epoch), path(name), path(jitter_offsets)
    
    script:
    name = "${epoch}.sampled_examples_mat.npz"
    jitter_offsets = "${epoch}.sampled_offsets.npz"
    """
    python3 $moduleDir/bin/sample_examples.py \
        ${meta.zarr_anndata} \
        ${params.sampling_dist} \
        ${name} \
        ${jitter_offsets} \
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
        | map(it -> tuple(it[3])) // meta epoch sampled_examples offsets

        //| extractDataWithOffsets

}
