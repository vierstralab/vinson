#!/bin/bash
#SBATCH --job-name=vinson_ag_predict
#SBATCH --output=/home/mbrannon/tmp/logs/vinson_ag_predict_%A_%a.out
#SBATCH --error=/home/mbrannon/tmp/logs/vinson_ag_predict_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpcg-test
#SBATCH --array=1-538

conda activate tangermeme
export MASTER_ADDR=127.0.0.1

# Make ag_list.txt from embeddings file columns

SCRIPT_PATH="/home/mbrannon/vinson/scripts/predict_cell.py"
OUTPUT_DIR="/home/mbrannon/tmp/ag_preds_mouse_nll"
EMBEDDINGS_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/mouse/embeddings.tsv"
READ_DEPTH_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/mouse/total_cutcounts.tsv"
FASTA_FILE="/net/seq/data/genomes/ENCODE3/mm10/male/mm10-encode3-male.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay_v2/checkpoints/epoch=3-step=1851994-val_loss=16.16.ckpt"
PEAK_DIR="/net/seq/data2/projects/ENCODE4Plus/hotspot3/mouse/output/"


mkdir -p "$OUTPUT_DIR"
mkdir -p "/home/mbrannon/tmp/logs_mouse"

# AG list file
AG_LIST="/home/mbrannon/tmp/mouse_ag_list.txt"

# pick the AG for this array index
ag=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $AG_LIST)

srun python $SCRIPT_PATH \
    --ag "$ag" \
    --fasta "$FASTA_FILE" \
    --embeddings "$EMBEDDINGS_FILE" \
    --read-depths "$READ_DEPTH_FILE" \
    --checkpoint "$MODEL_CKPT" \
    --peak-file-dir "$PEAK_DIR" \
    --output-dir "$OUTPUT_DIR"
    
