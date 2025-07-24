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

EVAL_SAMPLES_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/data_JUL10.batch2.val.h5"
EMBEDDINGS_FILE="/home/jvierstra/proj/vinson/data/embeddings.tsv"
READ_DEPTH_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/total_cutcounts.tsv"
FASTA_FILE="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/regression_data_JUL10.batch1_v3_poisson/checkpoints/last.ckpt"
OUTPUT_FILE="/home/mbrannon/output.tsv"


# Run the script
srun python /home/mbrannon/vinson/predict.py \
    --eval-samples "$EVAL_SAMPLES_FILE" \
    --embeddings "$EMBEDDINGS_FILE" \
    --read-depths "$READ_DEPTH_FILE" \
    --fasta "$FASTA_FILE" \
    --checkpoint "$MODEL_CKPT" \
    --output "$OUTPUT_FILE"