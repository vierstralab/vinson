#!/bin/bash
#SBATCH --job-name=vinson_predict
#SBATCH --output=/home/mbrannon/tmp/vinson_predict.out
#SBATCH --error=/home/mbrannon/tmp/vinson_predict.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpcg-test
#SBATCH --ntasks-per-node=8


conda activate tangermeme

export MASTER_ADDR=127.0.0.1

EVAL_SAMPLES_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/epoch_1/data_AUG3_pos.batch1.val.h5"
EMBEDDINGS_FILE="/home/jvierstra/proj/vinson/data/embeddings.tsv"
READ_DEPTH_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/total_cutcounts.tsv"
FASTA_FILE="/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa"
MODEL_CKPT="/home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay_v2/checkpoints/epoch=3-step=1851994-val_loss=16.16.ckpt"
OUTPUT_FILE="/home/mbrannon/tmp/AUG3_with_warmup_and_decay_prediction.tsv"
NEGATIVES_FILE="/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/epoch_1/data_AUG3_neg.batch1.val.bed.gz"
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
    --negatives "$NEGATIVES_FILE" \
    --output "$OUTPUT_FILE"