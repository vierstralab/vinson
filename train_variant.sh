#!/bin/bash -l
#SBATCH --partition=hpcg-test
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8 
#SBATCH --mem=256G

conda activate tf-2.13.0

export MASTER_ADDR=127.0.0.1

srun python /home/jvierstra/proj/vinson/train_variant_lit.py \
    --nodes 1 --devices 4 \
    --weights models/regression_data_JUL10.batch1_v4_poisson_test/checkpoints/epoch=5-step=1227796-val_loss=20.22.ckpt \
    --outdir /home/jvierstra/proj/vinson/models/variant_v3_normedloss/ \
    /home/jvierstra/proj/vinson/data/dnase-cavs.v4.train.h5 \
    /home/jvierstra/proj/vinson/data/dnase-cavs.v4.val.h5

