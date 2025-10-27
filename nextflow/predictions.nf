

process predict {
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(prefix), path(dhs_dataset), path(checkpoint), path(model_config), val(model_type)
    
    output:
        tuple val(prefix), path(dhs_dataset), path(name)
    
    script:
    name = "${prefix}.npy"
    """
    python3 $moduleDir/bin/predict_DHS_model.py \
        ${dhs_dataset} \
        ${params.zarr_anndata} \
        ${params.fasta_file} \
        --checkpoint ${checkpoint} \
        --genotype_file ${params.genotype_file} \
        --num_workers ${task.cpus} \
        --model_type ${model_type} \
        --output ${name} 
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

process visualize_predictions {
    tag "${prefix}"
    conda "${params.conda}"
    publishDir "${params.outdir}/nmf/${prefix}"
    label "med_mem"

    input:
        tuple val(prefix), path(dhs_dataset), path(predict_np)

    output:
        tuple val(prefix), path("*.pdf")

    script:
    """
    python3 $moduleDir/bin/plot_precomputed_data.py \
        --prefix ${prefix} \
        --dataset ${dhs_dataset} \
        --predict-output ${predict_np} \
        --output-dir ${params.outdir}
        
    """
}

workflow {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(row -> tuple(row.prefix, file(row.dhs_dataset), file(row.checkpoint), file(row.model_config), row.model_type))
        | predict
        | visualize_predictions
    
    annotate_with_predictions()
    
}
