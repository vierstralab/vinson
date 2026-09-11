
params.offsets = "-500:501:20"

process predict_variants {
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(meta), path(variant_dataset), val(sample_id)
    
    output:
        tuple val(meta), path(variant_dataset), path(name)
    
    script:
    prefix = meta.prefix
    name = "${prefix}.predictions_annotated.tsv"
    """
    python3 $moduleDir/bin/predict_variants.py \
        ${variant_dataset} \
        ${meta.zarr_anndata} \
        ${meta.fasta_file} \
        ${meta.checkpoint} \
        ${meta.model_config} \
        ${sample_id} \
        --output ${name} \
        --num_workers ${task.cpus} \
        --offsets ${params.offsets}
    """
}


workflow {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, file(it.variant_dataset), it.sample_id))
        | predict_variants

}