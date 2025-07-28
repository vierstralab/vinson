#!/bin/bash -l
#SBATCH --partition=gpuAll
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8 
#SBATCH --mem=256G

conda activate tf-2.13.0

export MASTER_ADDR=127.0.0.1

srun python /home/jvierstra/proj/vinson/train_lit.py \
    --nodes 1 --devices 4 \
    --regression \
    --outdir /home/jvierstra/proj/vinson/models/regression_data_JUL10.batch1_v5_poisson_genotypes/ \
    /net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/data_JUL10.batch1.train.h5 \
    /net/seq/data2/projects/sabramov/SuperIndex/hotspot3/w_babachi_new.v23/ml_prediction/JUL10/continious_annotation/data_JUL10.batch1.val.h5

