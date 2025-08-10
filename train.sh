#!/bin/bash -l
#SBATCH --partition=hpcg-test #gpuAll
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G

conda activate tf-2.13.0

export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=127.0.0.1

# if [[ $HOSTNAME == 'hpcg04-heavy' ]]; then
export NCCL_P2P_DISABLE=1
# fi

TRAIN_SAMPLES_FILES_PATTERN=/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/epoch_\*/data_AUG3_pos.batch1.train.h5
TRAIN_SAMPLES_NEG_FILES_PATTERN=/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/epoch_\*/data_AUG3_neg.batch1.train.bed.gz
VAL_SAMPLES_FILE=/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/epoch_1/data_AUG3_pos.batch1.val.h5
VAL_SAMPLES_NEG_FILE=/net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/AUG3/epoch_1/data_AUG3_neg.batch1.val.bed.gz

srun python /home/jvierstra/proj/vinson/train_lit.py \
     --regression \
     --nodes 1 --devices 8 \
     --accelerator gpu \
     --strategy ddp \
     --num_workers 8 \
     --batch_size 64 \
     --negative_weight 2 \
     --clip_density 4 \
     --outdir /home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay_restart \
     --checkpoint /home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay/checkpoints/epoch=5-step=2106188-val_loss=21.51.ckpt \
     "$TRAIN_SAMPLES_FILES_PATTERN" \
     "$TRAIN_SAMPLES_NEG_FILES_PATTERN" \
     $VAL_SAMPLES_FILE \
     $VAL_SAMPLES_NEG_FILE