#!/bin/bash
#SBATCH --job-name=vinson_predict
#SBATCH --output=vinson_predict.out
#SBATCH --error=vinson_predict.err
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --partition=pool
#SBATCH --ntasks-per-node=1

# Load modules or activate environment
source ~/.bashrc
conda activate tangermeme

export MASTER_ADDR=127.0.0.1

EVAL_SAMPLES_FILE="/home/mbrannon/tmp/eval_samples/1.h5"
EMBEDDINGS_FILE="/home/jvierstra/proj/vinson/data/embeddings.tsv"
FASTA_FILE="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/variant_data_JUL28_3_a100_clip_density/checkpoints/epoch=64-step=74880-val_loss=2.6043.ckpt"
OUTPUT_FILE="/home/mbrannon/tmp/test1.tsv"


# Run the script
srun python /home/mbrannon/vinson/predict_var.py \
    --eval-samples "$EVAL_SAMPLES_FILE" \
    --embeddings "$EMBEDDINGS_FILE" \
    --fasta "$FASTA_FILE" \
    --checkpoint "$MODEL_CKPT" \
    --output "$OUTPUT_FILE"