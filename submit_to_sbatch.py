#!/usr/bin/env python3
import subprocess
from datetime import datetime
import argparse
from pathlib import Path
import os

# Get the absolute path of the directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "template_submit.sbatch"

# run as 
# python submit_to_sbatch.py /net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/training_data/OCT22//epoch_1.h5ad /net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.fa /net/seq/data2/projects/sabramov/ENCODE4/dnase-wasp.v5/output/all_variants_stats.bed.gz /net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/vinson_model
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument("--cpus_per_gpu", type=int, default=4)
    parser.add_argument('anndata', type=str, help='Path to anndata file')
    parser.add_argument('fasta', type=str, help='Path to fasta file')
    parser.add_argument('genotype', type=str, help='Path to genotype file')
    parser.add_argument('outdir', type=str, help='Path to output directory')
    parser.add_argument('--preset', 
                        choices=('hpcg05-a100', 'hpcg04-heavy', 'hpcg01'), 
                        default=None, 
                        help='Preset sbatch parameters for different hpcg-test nodes. Overrides gpus_per_node and cpus_per_gpu if set.')

    args = parser.parse_args()

    cfg = dict(
        partition="hpcg-test",
        nodes=1,
        gpus_per_node=args.gpus_per_node,
        cpus_per_task=args.cpus_per_gpu,
        time="36:00:00",
        env_path="/home/sabramov/miniconda3/envs/pytorch",
        anndata=args.anndata,
        fasta=args.fasta,
        genotype=args.genotype,
        outdir=args.outdir,
        config="train_dhs_new_cluster_config.yaml",
    )


    if args.preset is not None:
        if args.preset == 'hpcg05-a100':
            cfg['gpus_per_node'] = 8
            cfg['cpus_per_task'] = 10
        elif args.preset == 'hpcg04-heavy':
            cfg['gpus_per_node'] = 10
            cfg['cpus_per_task'] = 3
        elif args.preset == 'hpcg01':
            cfg['gpus_per_node'] = 4
            cfg['cpus_per_task'] = 4


    # auto-generated bits
    cfg["run_name"] = subprocess.check_output(
        [f"{cfg['env_path']}/bin/python", "-c", "from vinson.utils import generate_run_name; print(generate_run_name())"],
        text=True
    ).strip()
    
    timestamp = datetime.now().strftime("%Y_%m_%d")
    cfg['run_name'] = f"{timestamp}_{cfg['run_name']}"
    # ---- render & submit ----
    with open(TEMPLATE_PATH) as f:
        script = f.read().format(**cfg)
    outdir = cfg['outdir'] + "/" + cfg['run_name']
    os.makedirs(outdir, exist_ok=True)

    script_path = f"{outdir}/submit.sh"
    with open(script_path, "w") as f:
        f.write(script)

    subprocess.run(["sbatch", script_path])
    print(f"Submitted run: {cfg['run_name']}")
    print(f"Outdir: {outdir}")
