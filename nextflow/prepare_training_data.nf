
params.sampling_dist = 0.38


process sample_examples {
    conda "${params.conda}"
    label "gpu"
    tag "${prefix}"

    input:
        tuple val(meta), val(epoch)
    
    output:
        tuple val(meta), path(epoch), path(name)
    
    script:
    name = "${epoch}.sampled_offsets.npy"
    """
    python3 $moduleDir/bin/sample_with_offsets \
        ${meta.zarr_anndata} \
        ${params.sampling_dist} \
        ${name}
    """
}



process save_epochs_data {
    conda "${params.conda}"

    input:
        tuple val(meta), val(epoch)

    output:
        path name

    script:
    name = "${epoch}.annotated_with_predictions.tsv"
    """
    python3 $moduleDir/bin/annotate_meta.py \
        ${samples_file} \
        ${base_path}/ \
        ${name}
    """
}

process assemble_train_anndata {

}


workflow {
    Channel.fromPath(params.samples_file)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, file(it.dhs_dataset)))
        | predict
        | visualize_cell_selective_predictions
    
    annotate_with_predictions(params.samples_file, "${params.outdir}")
    
}
