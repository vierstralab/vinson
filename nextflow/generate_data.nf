include { predict; annotate_with_predictions } from "./predictions"


process generate_sample_validation_data {
    conda "${params.conda}"
    publishDir "${params.outdir}/validation_data"
    tag "${prefix}"

    input:
        val meta

    output:
        tuple val(meta), path(name)

    script:
    prefix = "${meta.prefix}"
    name = "${prefix}.validation_data.h5"
    """
    python3 $moduleDir/bin/generate_validation_data.py \
        ${meta.zarr_anndata} \
        ${name} \
        --sample_ids ${meta.sample_id} \
        --mode ${meta.mode}
    """
}


process generate_dhs_validation_data {

    conda "${params.conda}"
    publishDir "${params.outdir}/validation_data"
    tag "${prefix}"

    input:
        val meta

    output:
        tuple val(meta), path(name)

    script:
    prefix = "${meta.prefix}"
    name = "${prefix}.validation_data.h5"
    """
    python3 $moduleDir/bin/generate_validation_data.py \
        ${meta.zarr_anndata} \
        ${name} \
        --dhs_ids ${meta.dhs_id}
    """
}

workflow {
    data = Channel.fromPath(params.validation_samples_file)
        | splitCsv(header:true, sep:'\t')
        | generate_sample_validation_data // meta, dhs_dataset
        | predict
    
    annotate_with_predictions(params.validation_samples_file)
}

workflow dhsValidation {
    Channel.fromPath(params.validation_dhs_file)
        | splitCsv(header:true, sep:'\t')
        | generate_dhs_validation_data
        | predict
    
    annotate_with_predictions(params.validation_dhs_file)
}


