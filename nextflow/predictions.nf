

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
    python3 $moduleDir/bin/predict_DHS_model.py \
        ${dhs_dataset} \
        ${meta.zarr_anndata} \
        ${meta.fasta_file} \
        ${meta.checkpoint} \
        ${meta.model_config} \
        ${genotype_args} \
        --num_workers ${task.cpus} \
        --batch_size ${params.prediction_batch_size} \
        --output ${name} 
    """
}


process visualize_cell_selective_predictions {
    tag "${prefix}"
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions/${prefix}"
    label "med_mem"

    input:
        tuple val(meta), path(dhs_dataset), path(predict_np)

    output:
        tuple val(meta), path("*.pdf")

    script:
    prefix = meta.prefix
    """
    python3 $moduleDir/bin/plot_cell_selective_data.py \
        --prefix ${prefix} \
        --h5_data ${dhs_dataset} \
        --npy_prediction ${predict_np} \
        --output ./ \
        --adata ${meta.zarr_anndata} \
        --train_adata ${params.train_anndata} \
        --annotation_data ${params.annotation_data} \
        --model_config ${meta.model_config}
    """
}


process annotate_with_predictions {
    conda "${params.conda}"
    publishDir "${params.outdir}/"

    input:
        path samples_file

    output:
        path name

    script:
    name = "${samples_file.baseName}.annotated_with_predictions.tsv"
    """
    python3 $moduleDir/bin/annotate_meta.py \
        ${samples_file} \
        ${params.outdir}/ \
        ${name}
    """
}


workflow {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, file(it.dhs_dataset)))
        | predict
        | visualize_cell_selective_predictions
    
    annotate_with_predictions(params.samples_file)
    
}

workflow visualize {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, file(it.dhs_dataset), file("${params.outdir}/predictions/${row.prefix}/${row.prefix}.npy")))
        | visualize_cell_selective_predictions
}


process predict_fasta {

    conda "${params.conda}"
    publishDir "${params.outdir}/fasta_predictions/${prefix}", pattern: "${name}"
    label "gpu"
    tag "${prefix}"

    input:
        val meta
    
    output:
        tuple val(meta), path(name)
    
    script:
    prefix = "${meta.prefix}"
    name = "${prefix}.npy"
    """
    echo "${meta.seq}" > fasta.tmp
    python3 $moduleDir/bin/predict_fasta.py \
        ${meta.prefix}
        fasta.tmp
        ${meta.zarr_anndata} \
        ${meta.checkpoint} \
        ${meta.model_config} \
        ${name}
    """

}

workflow predictFasta {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | predict_fasta
}