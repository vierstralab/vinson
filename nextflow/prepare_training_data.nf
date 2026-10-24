

include { extractDataWithOffsets } from "./pipelines/nf-index/extract_data_with_offsets" addParams(conda: params.conda_index)


params.sampling_dist = 0.38
params.jitter = 40


process sample_epoch {
    conda "${params.conda}"
    tag "${epoch}"
    publishDir "${params.outdir}/raw_matrices", pattern: "${sampled_examples}"
    cpus 10

    input:
        tuple val(epoch), val(meta)

    output:
        tuple val(meta), val(epoch), path(sampled_examples), path(sample_names)

    script:
    sampled_examples = "matrix.${epoch}.class.npz"
    sample_names = "${epoch}.sample_names.txt"
    """
    python3 $moduleDir/bin/sample_examples.py \
        ${meta.zarr_anndata} \
        ${params.sampling_dist} \
        ${sampled_examples} \
        ${sample_names} \
        --seed ${meta.seed}
    """
}


process jitter_offsets {
    conda "${params.conda}"
    tag "${epoch}"

    input:
        tuple val(meta), val(epoch), path(examples), path(sample_names)

    output:
        tuple val(meta), val(epoch), path(name), path(offsets_matrix)

    script:
    name = "${epoch}.sampled_offsets.npz"
    offsets_matrix = "matrix.${epoch}.offsets.npz"
    """
    python3 $moduleDir/bin/jitter_offsets.py \
        ${examples} \
        ${sample_names} \
        ${name} \
        ${offsets_matrix} \
        --seed ${meta.seed} \
        --jitter ${params.jitter}
    """
}

process assemble_train_anndata {
    conda "${params.conda}"
    publishDir "${params.outdir}/train_data"

    input:
        tuple val(zarr_anndata), val(prefix), path(matrices)

    output:
        tuple val(prefix), path(name)

    script:
    name = "${prefix}.anndata.h5ad"
    """
    python3 $moduleDir/bin/add_matrices_to_anndata.py \
        ${zarr_anndata} \
        ${name} \
        ${matrices} \
    """
}


// Variants skip sampling: class (pos/neg) is defined by filtering at anndata-creation
// time, not delta-ball sampling. All epochs reuse the same variants/reference (only
// the jitter seed differs per epoch), so the base anndata, its layers, and the class
// matrix are each built once and just replicated under every epoch's layer suffix.
// The class matrix is in the same on-disk convention as sample_epoch's output, so the
// existing jitter_offsets process can be reused unmodified to compute offsets per epoch.
process assemble_variant_train_anndata {
    conda "${params.conda}"
    publishDir "${params.outdir}/raw_matrices", pattern: "${class_matrix}"

    input:
        tuple val(epochs), path(variants_parquet), val(reference_anndata)

    output:
        tuple val(epochs), path(name), path(class_matrix), path(sample_names)

    script:
    name = "variant_anndata.zarr"
    class_matrix = "matrix.variants.class.npz"
    sample_names = "variant_samples.txt"
    """
    python3 $moduleDir/bin/variants/create_training_adata.py \
        ${variants_parquet} \
        ${reference_anndata} \
        ${name} \
        ${class_matrix} \
        ${sample_names} \
        --epoch ${epochs.join(' ')}
    """
}


workflow {
    input_data = Channel.fromPath(params.epochs_meta)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it.epoch, it))

    out = input_data
        | sample_epoch // meta, epoch, class matrix, sample_names
        | jitter_offsets
        | map(it -> tuple(it[1], it[2], it[0].zarr_anndata)) //epoch, offsets, anndata
        | extractDataWithOffsets
        | mix(sample_epoch.out.map(it -> tuple('class', it[2])))
        | mix(jitter_offsets.out.map(it -> tuple('offsets', it[3])))
        | combine(input_data)
        | map(it -> tuple(it[3].zarr_anndata, "train_dhs", it[1])) // prefix_or_class, matrix, epoch, meta -> zarr_anndata, prefix, matrix
        | unique()
        | groupTuple(by: [0, 1])
        | assemble_train_anndata
}


// run with: nextflow run prepare_training_data.nf -entry fromVariants
workflow fromVariants {
    input_data = Channel.fromPath(params.variant_epochs_meta)
        | splitCsv(header:true, sep:'\t')
        | map(it -> tuple(it, it.epoch)) // epoch, meta (meta.seed needed by jitter_offsets)

    // all epochs reuse the same variants_parquet + reference_anndata (only the jitter
    // seed differs), so build the base anndata + class matrix once for every epoch
    assembled = input_data
        | map(it -> tuple(it[1], file(it[0].variants_parquet), it[0].reference_anndata))
        | groupTuple(by: [1, 2]) // [epochs], variants_parquet, reference_anndata
        | assemble_variant_train_anndata // epochs, assembled_anndata, class_matrix, sample_names

    jittered = input_data
        | combine(assembled.map(it -> tuple(it[2], it[3]))) // meta, epoch, class_matrix, sample_names
        | jitter_offsets // meta, epoch, offsets.npz, offsets_matrix.npz

    // the variant anndata is masterlist-shaped (each variant is a 1bp "DHS")
    // extracts density at jittered position, like the DHS workflow.
    density_matrices = jittered
        | map(it -> tuple(it[1], it[2])) // epoch, offsets.npz
        | combine(assembled.map(it -> it[1])) // epoch, offsets.npz, zarr_anndata
        | extractDataWithOffsets // prefix(="{epoch}.{suffix}"), matrix
        | map(it -> it[1]) // matrix

    matrices = jittered
        | map(it -> it[3]) // offsets_matrix.npz
        | mix(density_matrices)
        | collect() // [offsets_matrix.npz, density/bg matrices, ...] once all epochs are done

    assembled
        | map(it -> tuple(it[1], "train_variant")) // zarr_anndata, prefix
        | combine(matrices) // zarr_anndata, prefix, [all matrix files]
        | assemble_train_anndata // reused: merges everything in and writes the final h5ad
}
