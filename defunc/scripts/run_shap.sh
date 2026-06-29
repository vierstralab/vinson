#!/bin/bash
#SBATCH --job-name=vinson_shap
#SBATCH --nodes=1
#SBATCH --output=/home/mbrannon/tmp/vinson_shap.out
#SBATCH --error=/home/mbrannon/tmp/vinson_shap.err
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpcg-test
#SBATCH --ntasks-per-node=8

conda activate tangermeme
export MASTER_ADDR=127.0.0.1

BED_FILE="/net/seq/data/dmz/www/data/encode4plus/encode-public/human/internal/by_aggregation/AG95390/AG95390.peaks.fdr0.001.bed.gz"
EMBEDDINGS_FILE="/home/jvierstra/proj/vinson/data/embeddings.tsv"
FASTA_FILE="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay_v2/checkpoints/epoch=3-step=1851994-val_loss=16.16.ckpt"
OUTPUT_FILE="/home/mbrannon/tmp/shapAG95390"
AG="AG95390"

# Run the script
srun python /home/mbrannon/vinson/scripts/shap_cell_type.py \
    --bed "$BED_FILE" \
    --embeddings "$EMBEDDINGS_FILE" \
    --fasta "$FASTA_FILE" \
    --model-file "$MODEL_CKPT" \
    --ag "$AG" \
    --output "$OUTPUT_FILE"