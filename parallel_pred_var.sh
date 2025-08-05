#!/bin/bash

EVAL_DIR="/home/mbrannon/tmp/eval_samples/"
OUTPUT_DIR="/home/mbrannon/tmp/variant_preds"
SCRIPT_PATH="/home/mbrannon/vinson/predict_var.py"

# Create output dir if missing
mkdir -p "$OUTPUT_DIR"

for i in {0..99}; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=vinson_predict_$i
#SBATCH --output=$OUTPUT_DIR/vinson_predict_$i.out
#SBATCH --error=$OUTPUT_DIR/vinson_predict_$i.err
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=1
#SBATCH --partition=pool
#SBATCH --ntasks-per-node=1

source ~/.bashrc
conda activate tangermeme

EVAL_SAMPLES_FILE="$EVAL_DIR/$i.h5"
EMBEDDINGS_FILE="/home/jvierstra/proj/vinson/data/embeddings.tsv"
READ_DEPTH_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/total_cutcounts.tsv"
FASTA_FILE="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/variant_data_JUL28_3_a100_clip_density/checkpoints/epoch=64-step=74880-val_loss=2.6043.ckpt"
OUTPUT_FILE="$OUTPUT_DIR/variant_pred_$i.tsv"

srun python $SCRIPT_PATH \\
    --eval-samples "\$EVAL_SAMPLES_FILE" \\
    --embeddings "\$EMBEDDINGS_FILE" \\
    --read-depths "\$READ_DEPTH_FILE" \\
    --fasta "\$FASTA_FILE" \\
    --checkpoint "\$MODEL_CKPT" \\
    --output "\$OUTPUT_FILE"
EOF
done
