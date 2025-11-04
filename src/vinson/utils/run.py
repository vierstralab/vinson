import anndata as ad

import torch

from vinson.models.sequence import CellEmbedding, BassetTrunkEmbed, EmbedModel
from vinson.lr import CosineAnnealingWarmupRestarts
from vinson.datamodules.sequence import SeqEmbedDataModule
from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.utils.data_formatting import extract_data_from_h5


def model_from_config(config, checkpoint_path=None):
    # TODO: add model configuration to config
    embed_model = CellEmbedding(n_inputs=637, n_layers=0, n_outputs=256)
    trunk_model = BassetTrunkEmbed(embed_model.n_outputs)

    if checkpoint_path is not None:
        model = EmbedModel.load_from_checkpoint(
            checkpoint_path,
            trunk=trunk_model,
            embed=embed_model,
        )
        return model

    # Create trunk model, maybe move to config later
    optimizer_kwargs = {
        'lr': config["hparams"]['lr']
    }
    model = EmbedModel(
        trunk=trunk_model,
        embed=embed_model,
        regression=config["model_type"] == "regression",
        lr_scheduler=config["hparams"]["lr_scheduler"],
        lr_scheduler_kwargs=config["hparams"]["lr_scheduler_kwargs"],
        optimizer_kwargs=optimizer_kwargs
        **config["model_kwargs"]
    )

    # Initialize model
    model.init_model()
    return model

############
def dataset_from_h5(
    h5_file: str,
    ref_adata: ad.AnnData,
    fasta_file: str,
    genotype_file: str = None,
    **dataset_kwargs
):
    """
    Initialize dataset.
    Args:
        h5_file (str): Path to the H5 file.
        fasta_file (str): Path to the FASTA file.
        genotype_file (str, optional): Path to the genotype file.
        **dataset_kwargs: Additional arguments for dataset.
    """
    data, embeddings_df = extract_data_from_h5(h5_file, ref_adata=ref_adata)
    dataset = SequenceEmbedDataset(
        data=data,
        embeddings_df=embeddings_df,
        fasta_file=fasta_file,
        genotype_file=genotype_file,
        **dataset_kwargs,
    )

    return dataset


def datamodule_from_config(
        config,
        anndata_file,
        fasta_file,
        genotype_file,
        **dataloader_kwargs
    ):
    """
    Initialize dataloaders.
    Args:
        config (dict): Configuration dictionary. See read_configs and default config for format.
        anndata_file (str): Path to the AnnData file.
        fasta_file (str): Path to the FASTA file.
        genotype_file (str): Path to the genotype file.
        batch_size (int): Batch size for dataloaders.
        **dataloader_kwargs: Additional arguments for dataloaders.
    """
    train_dataset_kwargs = {
        **config['data_params'],
        **config['train_augmentation_kwargs'],
    }

    valid_dataset_kwargs = {
        **config['data_params'],
        **config['validation_augmentation_kwargs'],
    }
    
    dataloader_kwargs = {
        'batch_size': config['hparams']['batch_size'],
        **dataloader_kwargs,
    }

    # DataModule to handle datasets updates and dataloader init
    return SeqEmbedDataModule(
        anndata_file=anndata_file,
        fasta_file=fasta_file,
        genotype_file=genotype_file,
        train_dataset_kwargs=train_dataset_kwargs,
        valid_dataset_kwargs=valid_dataset_kwargs,
        **dataloader_kwargs,
    )
