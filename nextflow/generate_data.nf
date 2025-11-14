
process generate_sample_validation_data {

    conda "${params.conda}"
    publishDir "${params.outdir}/validation_data"

    input:
        tuple val(sample_id), val(mode)

    output:
        tuple val(prefix), path(name)

    script:
    prefix = "${sample_id}.${mode}"
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
    Channel.fromPath(params.validation_samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> tuple(row.sample_id, row.mode))
        | generate_sample_validation_data
}

workflow dhsValidation {
    Channel.fromPath(params.validation_dhs_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> row.dhs_id)
        | generate_dhs_validation_data
}


