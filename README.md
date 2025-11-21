# vinson
ML sequence to function models


# Nextflow
## Predict sample DHSs
1. Generate samples_meta file (see ```nextflow/test_meta/test_per_sample_meta.tsv``` for the file format)
2. Edit `params.config` and `nextflow.config` according to your computational env specs.
3. run ```nextflow run generate_data.nf -profile Altius,new_cluster --validation_samples_file <your-meta-file> -resume```

## Predict cell-selective DHSs
1. Generate samples_meta file (see ```nextflow/test_meta/test_cell_selective_dhs_meta.tsv``` for the file format)
2. Edit `params.config` and `nextflow.config` according to your computational env specs.
3. run ```nextflow run predictions.nf -profile Altius,new_cluster --samples_file <your-meta-file> -resume```

