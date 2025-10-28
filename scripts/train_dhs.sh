#!/bin/bash -l
#SBATCH --partition=hpcg-test #gpuAll
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time 36:00:00

# srun --partition=hpcg-test --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=8 --mem=256G --pty bash

conda activate tf-2.13.0

# export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1

EMBEDDINGS_FILE=/home/jvierstra/proj/vinson/data/embeddings.tsv
FASTA_FILE=/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa

TRAIN_SAMPLES_FILES_PATTERN=/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/training_data/OCT8/epoch_\*/data_OCT8_pos.batch1.train.h5
TRAIN_SAMPLES_NEG_FILES_PATTERN=/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/training_data/OCT8/epoch_\*/data_OCT8_neg.batch1.train.bed.gz
VAL_SAMPLES_FILE=/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/training_data/OCT8/epoch_1/data_OCT8_pos.batch1.val.h5
VAL_SAMPLES_NEG_FILE=/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/training_data/OCT8/epoch_1/data_OCT8_neg.batch1.val.bed.gz

srun python /home/jvierstra/proj/vinson/scripts/train_dhs.py \
     --regression \
     --nodes 1 --devices 8 \
     --accelerator gpu \
     --strategy ddp \
     --num_workers 8 \
     --batch_size 32 \
     --negative_samples_rate 2 \
     --negative_weight 1 \
     --clip_density 20 \
     --min_bg 0.1 \
     --outdir /home/jvierstra/proj/vinson/models/data_OCT8_with_warmup_and_decay_v2 \
     --seed 42 \
     $EMBEDDINGS_FILE \
     $FASTA_FILE \
     "$TRAIN_SAMPLES_FILES_PATTERN" \
     "$TRAIN_SAMPLES_NEG_FILES_PATTERN" \
     $VAL_SAMPLES_FILE \
     $VAL_SAMPLES_NEG_FILE