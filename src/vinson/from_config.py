from vinson.models.sequence.lightning_wrappers import AbstractSequenceModel
from vinson.models.sequence.lightning_wrappers import SequenceEmbedModel, SequenceOnlyModel
from vinson.models.sequence.basset import BassetTrunk, BassetTrunkEmbed
from vinson.models.sequence.legnet import LegNetTrunk, LegNetTrunkEmbed

from vinson.models.variant.lightning_wrappers import VariantEmbedModel

from vinson.models.shared import MLPBlock


from vinson.datamodules.sequence import SeqEmbedDataModule, SeqEmbedVariantDataModule
from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.datasets.variant import VariantEmbedDataset

from vinson.utils.data_formatting import extract_data_from_h5

import anndata as ad


model_factory_registry = {
    "vinson": BassetTrunk,
    "legnet": LegNetTrunk,

    "vinson_embed": BassetTrunkEmbed,
    "legnet_embed": LegNetTrunkEmbed,
}

lightning_model_registry = {
    "vinson": SequenceOnlyModel,
    "legnet": SequenceOnlyModel,

    "vinson_embed": SequenceEmbedModel,
    "legnet_embed": SequenceEmbedModel,

    "vinson_variant_embed": VariantEmbedModel,
    "legnet_variant_embed": VariantEmbedModel,
}


def _parse_scheduler_and_optimizer(config):
    scheduler_name = config['hparams'].get("lr_scheduler")
    scheduler_kwargs = config['hparams'].get("lr_scheduler_kwargs", {})
    optimizer_kwargs = config['hparams']["optimizer_kwargs"]
    return {
        'lr_scheduler': scheduler_name,
        'lr_scheduler_kwargs': scheduler_kwargs,
        'optimizer_kwargs': optimizer_kwargs
    }


def dhs_model_from_config(config, checkpoint_path=None):
    model_type = config["model_type"]

    assert model_type in ('vinson', 'legnet', 'vinson_embed', 'legnet_embed'), f"Model type {model_type} not supported for DHS models."


    BaseModelCls = model_factory_registry[model_type]

    LightningModelCls: AbstractSequenceModel = lightning_model_registry[model_type]

    trunk = BaseModelCls(
        **config['base_arch']
    )

    head = MLPBlock(
        **config['head_arch']
    )

    torch_modules_kwargs = {
        "trunk_model": trunk,
        "head_model": head,
    }

    if model_type in ("vinson_embed", "legnet_embed"):
        mlp_embedding = MLPBlock(**config["cell_embed_arch"])
        torch_modules_kwargs["embed_model"] = mlp_embedding

    if checkpoint_path is not None:
        model = LightningModelCls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **torch_modules_kwargs,
        )
    
    else:
        scheduler_kwargs = _parse_scheduler_and_optimizer(config)
        model = LightningModelCls(
            **torch_modules_kwargs,
            **scheduler_kwargs,
            **config["model_kwargs"],
        )
    return model


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
