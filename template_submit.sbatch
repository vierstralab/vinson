#!/bin/bash -l
#SBATCH --job-name={run_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=400GB
#SBATCH --time={time}
#SBATCH --output={outdir}/{run_name}/slurm-%j.log

conda activate {env_path}

export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=127.0.0.1
export NCCL_P2P_DISABLE=1

echo "Run name: {run_name}"
echo "Output dir: {outdir}/{run_name}"

srun python $PWD/train_dhs.py \
    --run_name {run_name} \
    --nodes {nodes} \
    --accelerator gpu \
    --strategy ddp \
    --devices {gpus_per_node} \
    --num_workers {cpus_per_task} \
    --outdir {outdir} \
    --seed 42 \
    --genotype_file {genotype} \
    --config {config} \
    {anndata} \
    {fasta}
