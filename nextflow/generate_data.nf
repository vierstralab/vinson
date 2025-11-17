include { predict; annotate_with_predictions } from "./predictions"


process generate_sample_validation_data {

    conda "${params.conda}"
    publishDir "${params.outdir}/validation_data"
    tag "${prefix}"

    input:
        tuple val(prefix), val(sample_id), val(mode)

    output:
        tuple val(prefix), path(name)

    script:
    name = "${prefix}.validation_data.h5"
    """
    python3 $moduleDir/bin/generate_validation_data.py \
        ${params.zarr_anndata} \
        ${name} \
        --sample_ids ${sample_id} \
        --mode ${mode}
    """
}


process generate_dhs_validation_data {

    conda "${params.conda}"
    publishDir "${params.outdir}/validation_data"

    input:
        val dhs_id

    output:
        tuple val(prefix), path(name)

    script:
    prefix = "${dhs_id}"
    name = "${prefix}.validation_data.h5"
    """
    python3 $moduleDir/bin/generate_validation_data.py \
        ${params.zarr_anndata} \
        ${name} \
        --dhs_ids ${dhs_id}
    """
}

workflow {
    meta = Channel.fromPath(params.validation_samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> tuple(
            row.prefix,
            row.sample_id,
            row.mode,
            file(row.checkpoint),
            row.model_config,
            row.model_type
            )
        )
    
    meta
        | map(it -> tuple(it[0], it[1], it[2]))
        | generate_sample_validation_data // prefix, dhs_dataset
        | join(meta.map(it -> tuple(it[0], it[3], it[4], it[5]))) // prefix, dhs_dataset, checkpoint, model_config, model_type
        | predict
    
    annotate_with_predictions(params.validation_samples_file)
}

workflow dhsValidation {
    Channel.fromPath(params.validation_dhs_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> row.dhs_id)
        | generate_dhs_validation_data
    
    //annotate_with_predictions(params.validation_samples_file)
}


