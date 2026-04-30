


process predict_variants {
    conda "${params.conda}"
    publishDir "${params.outdir}/predictions"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(meta), path(variant_dataset)
    
    output:
        tuple val(meta), path(variant_dataset), path(name)
    
    script:
    prefix = meta.prefix
    name = "${prefix}.predictions_annotated.tsv"
    """
    python3 $moduleDir/bin/predict_DHS_model.py \
        ${variant_dataset} \
        ${meta.zarr_anndata} \
        ${meta.fasta_file} \
        ${meta.checkpoint} \
        ${meta.model_config} \
        ${name}
    """
}


workflow {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, file(it.variant_dataset)))
        | predict_variants
 

}