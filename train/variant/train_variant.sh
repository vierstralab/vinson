#!/bin/bash -l
#SBATCH --partition=hpcg-test
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=4 
#SBATCH --mem=128G

module load cuda-toolkit/nccl_2.27.5-1
conda activate tf-2.13.0

# export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1


TRUNK_WEIGHTS="/home/jvierstra/proj/vinson/models/data_AUG3_with_warmup_and_decay_v2/checkpoints/last.ckpt"
TRAIN_SAMPLES_FILE="/home/jvierstra/proj/vinson/data/dnase-cavs.v4.train.var-split.h5"
VAL_SAMPLES_FILE="/home/jvierstra/proj/vinson/data/dnase-cavs.v4.val.var-split.h5"

srun python /home/jvierstra/proj/vinson/scripts/train_variant.py \
    --nodes 1 --devices 4 \
    --accelerator gpu \
    --strategy ddp \
    --num_workers 4 \
    --batch_size 32 \
    --trunk_weights $TRUNK_WEIGHTS \
    --outdir /home/jvierstra/proj/vinson/models/variant_data_AUG3_with_warmup_and_decay_v2 \
    $TRAIN_SAMPLES_FILE \
    $VAL_SAMPLES_FILE

