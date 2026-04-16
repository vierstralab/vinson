import csv
import os
import sys
import random
import numpy as np
import anndata as ad
from argparse import ArgumentParser

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from vinson.utils.helpers import save_config, generate_run_name
from vinson.run import set_global_seed,set_worker_seed,init_multigpu_trainer, fit_model
from vinson.from_config import read_configs,datamodule_from_config,variant_model_from_config
from vinson.utils.data_formatting.adata_utils import get_number_of_train_examples


torch.set_float32_matmul_precision('high')

def main(args):
    run_name = args.run_name.strip() or "vinson"
    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)
    epochs=args.epochs

    prev_run_config = os.path.join(outdir, "run_config.yaml")
    if os.path.exists(prev_run_config) and args.config is None:
        print(
            "Found existing config in output directory and no custom config provided. "
            "Using existing config."
        )
        args.config = prev_run_config
    #use other default for legnet
    default_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "default_train_variant_legnet.config.yaml"
    )
    config = read_configs(
        default_config_path,
        overwrite_config_path=args.config,
    )

    config["command"] = " ".join(["python"] + sys.argv)
    config_path = os.path.join(outdir, "run_config.yaml")
    save_config(config, config_path)
    
    set_global_seed(args.seed)
    
    checkpoint = (
        os.path.join(outdir, "checkpoints", "last.ckpt")
        if args.checkpoint == "last"
        else args.checkpoint
    )
    trainer_kwargs = config.get("trainer_kwargs", {})
    
    if args.debug:
        epochs = 5
        trainer_kwargs["limit_train_batches"] = 100 * args.devices
        trainer_kwargs["limit_val_batches"] = 100 * args.devices
        config["logging_params"]["val_check_interval"] = 1.0
    
    early_stopping = config['hparams']['lr_scheduler'] != 'OneCycleLR'
        
    trainer = init_multigpu_trainer(
        outdir,
        accelerator=args.accelerator,
        strategy="ddp_find_unused_parameters_true",
        nodes=args.nodes,
        devices=args.devices,
        early_stopping=early_stopping,
        logger_type=config["logging_params"]["logger_type"],
        val_check_interval=config["logging_params"]["val_check_interval"],
        max_epochs = epochs,
        **trainer_kwargs, #should include max_epochs if want
    )
    
    dataloader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True if args.accelerator == "gpu" else False,
        drop_last=True,
        worker_init_fn=set_worker_seed,
        persistent_workers=False
    )

    # Setup dataloaders -- adjust to have correct input for variant
    datamodule = datamodule_from_config(
        config,
        anndata_file=args.anndata_file,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        **dataloader_kwargs,
    )
    
    datamodule.setup(stage="fit") 

    model = variant_model_from_config(config, sequence_model_checkpoint=args.sequence_model_checkpoint, checkpoint_path=checkpoint)

    
    if config['hparams']['lr_scheduler'] == 'OneCycleLR':
        if config['hparams']['lr_scheduler_kwargs'].get('total_steps') is None:
            print('Setting total_steps for OneCycleLR...')
            n_examples = get_number_of_train_examples(args.anndata_file,layer_prefix="ref_counts")
            print(f"n_examples: {n_examples}")
            config['hparams']['lr_scheduler_kwargs']['total_steps'] = round(n_examples / datamodule.dataloader_kwargs['batch_size'] / trainer.num_devices)
        else:
            print('total steps is in config')

    fit_model(
        model,
        trainer,
        datamodule,
        checkpoint=checkpoint,
    )
    

if __name__ == "__main__":
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument("anndata_file", type=str, help="Input AnnData file.")
    parser.add_argument("fasta_file", type=str, help="FASTA file")

    # Optional arguments
    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Unique identifier for the training run. Generated if not provided.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (see default config for format). "
        "If provided, overrides default parameters.",
    )
    parser.add_argument(
        "--genotype_file",
        type=str,
        default=None,
        help="Path to Tabix indexed genotype file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint or 'last' to resume last checkpoint.",
    )
    parser.add_argument(
        "--sequence_model_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load weights from dhs model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for logs and checkpoints.",
    )

    # --- Multi-GPU setup ---
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--devices", type=int, default=4, help="GPUs/CPUs per node.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Type of accelerator (e.g., 'gpu', 'cpu').",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Distributed training strategy (e.g., 'ddp', 'auto').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with limited training steps per epoch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of worker processes for data loading.",
    )

    args = parser.parse_args()
    main(args)