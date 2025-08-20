#!/bin/bash
#SBATCH --job-name=vinson_predict
#SBATCH --output=vinson_predict.out
#SBATCH --error=vinson_predict.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=pool
#SBATCH --ntasks-per-node=4

# Load modules or activate environment
source ~/.bashrc
conda activate tangermeme

export MASTER_ADDR=127.0.0.1

EVAL_SAMPLES_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/data_AUG3.val.heldout_samples.h5"
EMBEDDINGS_FILE="/home/jvierstra/proj/vinson/data/embeddings.tsv"
READ_DEPTH_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/total_cutcounts.tsv"
FASTA_FILE="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay/checkpoints/epoch=5-step=2106188-val_loss=21.51.ckpt"
OUTPUT_FILE="/home/mbrannon/tmp/AUG3_with_warmup_and_decay_prediction.tsv"
SAMPLE_GENOTYPE_FILE="/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v4/output/meta+sample_ids.tsv"
GENOTYPE_FILE="/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v4/output/all_variants_stats.bed.gz"



# Run the script
srun python /home/mbrannon/vinson/predict.py \
    --eval-samples "$EVAL_SAMPLES_FILE" \
    --embeddings "$EMBEDDINGS_FILE" \
    --read-depths "$READ_DEPTH_FILE" \
    --fasta "$FASTA_FILE" \
    --checkpoint "$MODEL_CKPT" \
    --sample-genotype-file "$SAMPLE_GENOTYPE_FILE" \
    --genotype-file "$GENOTYPE_FILE" \
    --output "$OUTPUT_FILE"