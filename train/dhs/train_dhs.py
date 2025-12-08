import os
import sys
import random
import numpy as np
from argparse import ArgumentParser

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from vinson.utils.helpers import read_configs, save_config, generate_run_name
from vinson.utils.run import datamodule_from_config, model_from_config


torch.set_float32_matmul_precision('high')


def set_global_seed(seed=42):
    # Python's built-in random module
    random.seed(seed)

    # Numpy's random module
    np.random.seed(seed)

    # PyTorch seed for CPU
    torch.manual_seed(seed)

    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)

    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id):
    # Set seed for Python and NumPy in each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# class IterateDataModule(L.Callback):
#     def on_train_epoch_end(self, trainer, pl_module):
#         trainer.datamodule.iterate_train_dataset()


def init_multigpu_trainer(
        outdir,
        accelerator,
        strategy,
        nodes,
        devices,
        logger_type,
        val_check_interval,
        **trainer_kwargs
    ):
    assert logger_type in ["csv"], "Only 'csv' logger is currently supported."
    logger = CSVLogger(os.path.join(outdir, "logs"))

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.005, patience=10),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.2f}",
            dirpath=os.path.join(
                outdir,
                "checkpoints",
            ),
            save_top_k=5,
            save_last="link",
        ),
        LearningRateMonitor(),
    ]

    trainer = L.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=20,
        accelerator=accelerator,
        strategy=strategy,
        num_nodes=nodes,
        devices=devices,
        val_check_interval=val_check_interval,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        **trainer_kwargs,
       # precision='16-mixed',
    )
    return trainer


def fit_model(model, trainer: L.Trainer, datamodule, checkpoint=None):
    if checkpoint is not None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
    else:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "anndata_file",
        type=str,
        help="Input AnnData file.",
    )

    parser.add_argument(
        "fasta_file", type=str, help="FASTA file",
    )

    parser.add_argument(
        "--run_name",
        default=None,
        type=str,
        help="Unique identifier for the training run. Generated if not provided.",
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None, 
        help='Path to YAML config file (see default config for format). If provided, overrides default parameters.'
    )

    parser.add_argument(
        "--genotype_file",
        type=str,
        default=None,
        help="Path to Tabix indexed genotype file.",
    )

    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint.", default=None,
    )

    parser.add_argument(
        "--seed", type=int, help="Random seed", default=42,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output directory for logs and checkpoints.",
    )
    # --- torch multi-gpu setup ---

    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes for distributed training."
    )
    parser.add_argument(
        "--devices", type=int, default=4, help="Number of devices (GPUs/CPUs) per node."
    )

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
        help="Type of accelerator to use (e.g., 'gpu', 'cpu').",
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
        help="Enable debug mode with max 200 * devices training steps per epoch.",
    )

    args = parser.parse_args()

    # Setup output
    run_name = args.run_name.strip() or "vinson" #or generate_run_name() generates unique name in each subprocess. currently done outside of this script

    outdir = os.path.join(args.outdir, run_name)
    os.makedirs(outdir, exist_ok=True)
    
    prev_run_config = os.path.join(outdir, "run_config.yaml")

    if os.path.exists(prev_run_config) and args.config is None:
        print("Found existing config in output directory and no custom config provided. Using existing config.")
        args.config = prev_run_config

    default_config_path = os.path.dirname(os.path.abspath(__file__)) + "/default_train_dhs.config.yaml"

    config = read_configs(
        default_config_path,
        custom_config_path=args.config
    )
    config['command'] = " ".join(["python"] + sys.argv)
    config_path = os.path.join(outdir, "run_config.yaml")
    save_config(
        config,
        config_path,
    )

    # Set global seed
    set_global_seed(args.seed)

    if args.checkpoint == "last":
        checkpoint = os.path.join(outdir, "checkpoints", "last.ckpt")
    else:
        checkpoint = args.checkpoint
    # Initialize model from config
    trainer_kwargs = {}

    if args.debug:
        trainer_kwargs['limit_train_batches'] = 200 * args.devices
        trainer_kwargs['limit_val_batches'] = 200 * args.devices
        config["logging_params"]["val_check_interval"] = 1.0

    # Initialize trainer
    trainer = init_multigpu_trainer(
        outdir,
        accelerator=args.accelerator,
        strategy=args.strategy,
        nodes=args.nodes,
        devices=args.devices,
        logger_type=config["logging_params"]["logger_type"],
        val_check_interval=config["logging_params"]["val_check_interval"],
        **trainer_kwargs
    )

    dataloader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True if args.accelerator == "gpu" else False,
        drop_last=True,
        worker_init_fn=set_worker_seed,
        persistent_workers=False
    )

    # Setup dataloaders
    datamodule = datamodule_from_config(
        config,
        anndata_file=args.anndata_file,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        **dataloader_kwargs,
    )
    
    if config['hparams']['lr_scheduler'] == 'OneCycleLR':
        if config['hparams']['lr_scheduler_kwargs']['total_steps'] is None:
            print('Setting total_steps for OneCycleLR...')
            datamodule.setup('fit')
            config['hparams']['lr_scheduler_kwargs']['total_steps'] = config['hparams']['epochs'] * datamodule.define_steps_per_epochs()

    model = model_from_config(config, checkpoint_path=checkpoint)

    # Start training
    fit_model(
        model,
        trainer,
        datamodule,
        checkpoint=checkpoint,
    )
