#!/bin/bash -l
#SBATCH --partition=hpcg-test #gpuAll
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time 36:00:00

# srun --partition=hpcg-test --nodes=1 --gres=gpu:2 --ntasks-per-node=2 --cpus-per-task=8 --mem=256G --pty bash

conda activate /home/sabramov/miniconda3/envs/pytorch

# export NCCL_DEBUG=INFO
export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1

ANNDATA=/net/seq/data2/projects/ENCODE4Plus/REGULOME/one_big_beautiful_index/latest.reference_anndata.zarr
FASTA_FILE=/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa
GENOTYPE_FILE=/net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/output/all_variants_stats.bed.gz

OUTDIR=/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/vinson_model/

# might need to change config path. path is relative to PWD now
srun python /home/jvierstra/proj/vinson/train_dhs.py \
    --nodes 1 \
    --accelerator gpu \
    --strategy ddp \
    --devices 8 \
    --num_workers 4 \
    --outdir $OUTDIR \
    --seed 42 \
    --genotype_file $GENOTYPE_FILE \
    --config $PWD/train_dhs_new_cluster_config.yaml \
    $ANNDATA \
    $FASTA_FILE \