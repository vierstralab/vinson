

process predict {
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions/${prefix}", pattern: "${name}"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(prefix), path(dhs_dataset), path(checkpoint), path(model_config), val(model_type)
    
    output:
        tuple val(prefix), path(dhs_dataset), path(model_config), path(name)
    
    script:
    name = "${prefix}.npy"
    """
    python3 $moduleDir/bin/predict_DHS_model.py \
        ${dhs_dataset} \
        ${params.zarr_anndata} \
        ${params.fasta_file} \
        ${checkpoint} \
        ${model_config} \
        --genotype_file ${params.genotype_file} \
        --num_workers ${task.cpus} \
        --model_type ${model_type} \
        --batch_size ${params.prediction_batch_size} \
        --output ${name} 
    """
}


process visualize_predictions {
    tag "${prefix}"
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions/${prefix}"
    label "med_mem"

    input:
        tuple val(prefix), path(dhs_dataset), val(model_config), path(predict_np)

    output:
        tuple val(prefix), path("*.pdf")

    script:
    """
    python3 $moduleDir/bin/plot_precomputed_data.py \
        --prefix ${prefix} \
        --h5_data ${dhs_dataset} \
        --npy_prediction ${predict_np} \
        --output ./ \
        --adata ${params.zarr_anndata} \
        --annotation_data ${params.annotation_data} \
        --model_config ${model_config}
    """
}


process annotate_with_predictions {
    conda "${params.conda}"
    publishDir "${params.outdir}/"
    label "ldsc"

    output:
        path name

    script:
    name = "${file(params.samples_file).baseName}.annotated_with_predictions.tsv"
    """
    python3 $moduleDir/bin/annotate_meta.py \
        ${params.samples_file} \
        ${params.outdir}/predictions \
        ${name}
    """
}

workflow test {
    models_data = Channel.fromPath(params.test_samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> tuple(file(row.checkpoint), file(row.model_config), row.model_type))
    
    generate_cell_selective_data()
        | combine(models_data)
        | predict
        | visualize_predictions
}


workflow {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> tuple(row.prefix, file(row.dhs_dataset), file(row.checkpoint), row.model_config, row.model_type))
        | predict
        | visualize_predictions
    
    annotate_with_predictions()
    
}

workflow visualize {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> tuple(row.prefix, file(row.dhs_dataset), file(row.model_config), file("${params.outdir}/predictions/${row.prefix}/${row.prefix}.npy")))
        | visualize_predictions
}
