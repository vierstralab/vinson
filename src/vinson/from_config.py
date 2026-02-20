from typing import Union

import anndata as ad

from datetime import datetime
import mergedeep

import torch
from vinson.utils.helpers import read_yaml_config

from vinson.models.sequence.lightning_wrappers import AbstractSequenceModel
from vinson.models.sequence.lightning_wrappers import SequenceEmbedModel, SequenceOnlyModel
from vinson.models.sequence.basset import BassetTrunk, BassetTrunkEmbed
from vinson.models.sequence.legnet import LegNetTrunk, LegNetTrunkEmbed

from vinson.models.variant.lightning_wrappers import VariantEmbedModel

from vinson.models.cell_classifier import CellClassifierModel

from vinson.models.shared import MLPBlock


from vinson.datamodules.sequence import SeqEmbedDataModule
from vinson.datamodules.variant import SeqEmbedVariantDataModule
from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.datasets.variant import VariantEmbedDataset

from vinson.utils.data_formatting import extract_data_from_h5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_FACTORY_REGISTRY = {
    "basset": BassetTrunk,
    "legnet": LegNetTrunk,

    "basset_embed": BassetTrunkEmbed,
    "legnet_embed": LegNetTrunkEmbed,

    "basset_variant_embed": BassetTrunkEmbed,
    "legnet_variant_embed": LegNetTrunkEmbed,
}

LIGHTNING_MODEL_REGISTRY = {
    "basset": SequenceOnlyModel,
    "legnet": SequenceOnlyModel,

    "basset_embed": SequenceEmbedModel,
    "legnet_embed": SequenceEmbedModel,

    "basset_variant_embed": VariantEmbedModel,
    "legnet_variant_embed": VariantEmbedModel,
}

def read_configs(config_path, overwrite_config_path=None):
    config = read_yaml_config(config_path)
    if overwrite_config_path is not None:
        update_config = read_yaml_config(overwrite_config_path)
        mergedeep.merge(config, update_config, strategy=mergedeep.Strategy.REPLACE)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return config


def _parse_scheduler_and_optimizer(config: dict):
    scheduler_name = config['hparams'].get("lr_scheduler")
    scheduler_kwargs = config['hparams'].get("lr_scheduler_kwargs", {})
    optimizer_kwargs = config['hparams']["optimizer_kwargs"]
    return {
        'lr_scheduler': scheduler_name,
        'lr_scheduler_kwargs': scheduler_kwargs,
        'optimizer_kwargs': optimizer_kwargs
    }


def classifier_model_from_config(config: dict, checkpoint_path: str = None):
    assert config['model_type'] == "classifier"
    embedding = MLPBlock(
        **config['model_arch']['cell_embed']
    )

    if checkpoint_path is not None:
        model = CellClassifierModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            embedding=embedding,
            map_location=device
        )
    else:
        scheduler_kwargs = _parse_scheduler_and_optimizer(config)
        model = CellClassifierModel(
            embedding=embedding,
            **scheduler_kwargs,
            **config.get('model_kwargs', {}),
        )
    return model


def _sequence_model_from_config(config):
    model_type = config["model_type"]
    BaseModelCls = MODEL_FACTORY_REGISTRY[model_type]

    if model_type in ("basset_embed", "legnet_embed", "basset_variant_embed", "legnet_variant_embed"):
        cell_embed_config = config['model_arch']['cell_embed']

        n_independent_embeds = config['model_arch'].get('n_independent_embeds', 1)
        if n_independent_embeds == 1:
            embeddings_mlp = MLPBlock(**cell_embed_config)
        else:
            embeddings_mlp = [
                MLPBlock(**config['model_arch']['cell_embed']) for _ in range(n_independent_embeds)
            ]

        config['model_arch']['trunk']['embeds'] = embeddings_mlp

    trunk = BaseModelCls(
        **config['model_arch']['trunk']
    )

    if hasattr(trunk, 'output_dim'):
        if config['model_arch']['head']['n_inputs'] is None:
            config['model_arch']['head']['n_inputs'] = trunk.output_dim

    return trunk


def dhs_model_from_config(config, checkpoint_path=None):
    model_type = config["model_type"]

    assert model_type in MODEL_FACTORY_REGISTRY, f"Model type {model_type} not supported for DHS models. Available types: {list(MODEL_FACTORY_REGISTRY.keys())}"

    LightningModelCls: AbstractSequenceModel = LIGHTNING_MODEL_REGISTRY[model_type]

    trunk = _sequence_model_from_config(config)

    head = MLPBlock(
        **config['model_arch']['head']
    )

    if checkpoint_path is not None:
        model = LightningModelCls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            trunk_model=trunk,
            head_model=head,
            map_location=device
        )
    
    else:
        scheduler_kwargs = _parse_scheduler_and_optimizer(config)
        model = LightningModelCls(
            trunk_model=trunk,
            head_model=head,
            **scheduler_kwargs,
            **config["model_kwargs"],
        )
    return model


def variant_model_from_config(config, sequence_model_checkpoint=None, checkpoint_path=None):
    model_type = config["model_type"]
    assert model_type in ("basset_variant_embed", "legnet_variant_embed"), f"Model type {model_type} not supported for variant models. Available types: 'basset_variant_embed', 'legnet_variant_embed'"

    head = MLPBlock(
        **config['model_arch']['head']
    )

    scheduler_kwargs = _parse_scheduler_and_optimizer(config)

    if checkpoint_path is not None:
        if sequence_model_checkpoint is not None:
            print('Ignoring sequence_model_checkpoint, as variant models load from a single checkpoint_path.')
        

    if sequence_model_checkpoint is not None and checkpoint_path is None:
        sequence_model = dhs_model_from_config(
            config,
            checkpoint_path=sequence_model_checkpoint,
        )
        variant_model = VariantEmbedModel.from_sequence_embed_model(
            sequence_embed_model=sequence_model,
            head_model=head,
            **scheduler_kwargs,
            **config["model_kwargs"],
        )
    else:
        trunk = _sequence_model_from_config(config)


        if checkpoint_path is not None:
            variant_model = VariantEmbedModel.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                trunk_model=trunk,
                head_model=head,
                map_location=device
            )
        else:
            variant_model = VariantEmbedModel(
                trunk_model=trunk,
                head_model=head,
                **scheduler_kwargs,
                **config["model_kwargs"],
            )
    return variant_model



def dataset_from_h5(
    h5_file: str,
    fasta_file: str,
    config,
    ref_adata: ad.AnnData = None,
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
    if config["model_type"] in ('vinson_variant_embed', 'legnet_variant_embed'):
        data = extract_data_from_h5(h5_file, ref_adata=ref_adata, is_variant=True)
        dataset = VariantEmbedDataset(
            data=data,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            **dataset_kwargs,
        )
    else:
        data = extract_data_from_h5(h5_file, ref_adata=ref_adata, is_variant=False)
        dataset = SequenceEmbedDataset(
            data=data,
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
    if config["model_type"] in ('vinson_variant_embed', 'legnet_variant_embed'):
        return SeqEmbedVariantDataModule(
            anndata_file=anndata_file,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            train_dataset_kwargs=train_dataset_kwargs,
            valid_dataset_kwargs=valid_dataset_kwargs,
            **dataloader_kwargs,
            )
    else:
        return SeqEmbedDataModule(
            anndata_file=anndata_file,
            fasta_file=fasta_file,
            genotype_file=genotype_file,
            train_dataset_kwargs=train_dataset_kwargs,
            valid_dataset_kwargs=valid_dataset_kwargs,
            **dataloader_kwargs,
        )
