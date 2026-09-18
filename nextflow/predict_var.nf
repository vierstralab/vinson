process predict {
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions/${prefix}", pattern: "${name}"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(meta), path(dhs_dataset)
    
    output:
        tuple val(meta), path(dhs_dataset), path(name)
    
    script:
    prefix = meta.prefix
    name = "${prefix}.npy"
    genotype_args = meta?.genotype_file ? "--genotype_file ${meta.genotype_file}" : ""
    """
    python3 $moduleDir/bin/predict_variant_model.py \
        ${meta.zarr_anndata} \
        ${meta.fasta_file} \
        ${meta.checkpoint} \
        ${meta.model_config} \
        ${genotype_args} \
        --h5_file ${dhs_dataset} \
        --num_workers ${task.cpus} \
        --batch_size ${params.prediction_batch_size} \
        --output ${name} 
    """
}
process concat_by_cell_type {

    conda "${params.conda}"

    publishDir "${params.outdir}/cell_type_results"

    tag "${cell_type}"

    input:
        tuple val(cell_type), path(pred_files)

    output:
        path "${cell_type}_all_variants.tsv"

    script:

    """
    python3 $moduleDir/bin/concat_results.py \
        ${cell_type} \
        ${cell_type}_all_variants.tsv \
        ${pred_files}
    """
}

process combine_predictions {

    conda "${params.conda}"

    publishDir "${params.outdir}/group_results"

    tag "${meta.prefix}"

    input:
        tuple val(meta), path(h5_file), path(pred_file)

    output:
        path "${meta.prefix}.parquet"

    script:

    """
    python3 $moduleDir/bin/combine_predictions.py \
        ${h5_file} \
        ${pred_file} \
        ${meta.prefix} \
        ${meta.prefix}.parquet
    """
}

process aggregate_group {

    conda "${params.conda}"

    publishDir "${params.outdir}/group_results"

    tag "${meta.prefix}"

    input:
        tuple val(meta), path(h5_file), path(pred_file)
        val aggregation_type
        path annotation_file

    output:
        path "${meta.prefix}.parquet"

    script:

    """
    python3 $moduleDir/bin/aggregate.py \
        ${h5_file} \
        ${pred_file} \
        ${meta.prefix} \
        ${meta.prefix}.parquet \
        ${aggregation_type} \
        ${annotation_file}
    """
}

process concat_results {

    conda "${params.conda}"

    publishDir "${params.outdir}"

    input:
        path parquet_files

    output:
        path "all_groups_with_predictions_wmeta_agg.parquet"

    """
    python3 $moduleDir/bin/concat_results.py \
        all_groups_with_predictions_wmeta_agg.parquet \
        ${parquet_files}
    """
}


workflow {
    predictions = Channel.fromPath(params.samples_file)
            | splitCsv(header:true, sep:'\t')
            | map { tuple(it, file(it.dhs_dataset)) }
            | predict \
            | combine_predictions \
            | collect \
            | concat_results
}

workflow aggregate {

    predictions = Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map { tuple(it, file(it.dhs_dataset)) }
        | predict

    aggregated = aggregate_group(
        predictions,
        params.aggregation_type,
        file(params.tested_annotation_file)
    )

    aggregated
        | collect
        | concat_results
}

workflow cell_type {

    predictions = Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map { tuple(it, file(it.dhs_dataset)) }
        | predict
        | combine_predictions

    predictions
        .map { meta, h5_file, pred_file ->
            tuple(meta.celltype, pred_file)
        }
        .groupTuple()
        | concat_by_cell_type
}
