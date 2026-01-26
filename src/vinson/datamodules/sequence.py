import lightning.pytorch as L
import numpy as np
from torch.utils.data import DataLoader
from vinson.datasets.sequence import SequenceEmbedDataset
from vinson.utils.data_formatting import extract_data_from_train_anndata, extract_variant_data_from_anndata
from vinson.datasets.sequence import SequenceEmbedDataset, VariantEmbedDataset
from itertools import cycle
from torchdata.stateful_dataloader import StatefulDataLoader

import anndata as ad
from tqdm import tqdm
import gc

# TODO: move all logging to one helper file
import logging
logger = logging.getLogger(__name__)


class SeqEmbedDataModule(L.LightningDataModule):
    def __init__(
        self,
        anndata_file: str,
        fasta_file: str,
        genotype_file=None,
        train_dataset_kwargs={},
        valid_dataset_kwargs={},
        **dataloader_kwargs,
    ):
        """
        Initialize the SeqEmbedDataModule.

        Args:
            adata (ad.AnnData): AnnData object containing the dataset.
            fasta_file (str): Path to the FASTA file.
            genotype_file (str, optional): Path to the genotype file.
            variant (bool): determines if varaiant dataset or sequence dataset
            train_dataset_kwargs (dict, optional): Additional arguments for the training dataset.
            valid_dataset_kwargs (dict, optional): Additional arguments for the validation dataset.
            dataloader_kwargs (dict, optional): Additional arguments for train and validation dataloaders. See DataLoader.__init__ for options.
        """
        super().__init__()

        self.fasta_file = fasta_file
        self.anndata_file = anndata_file

        self.genotype_file = genotype_file

        self.train_dataset_kwargs = train_dataset_kwargs
        self.valid_dataset_kwargs = valid_dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        self.train_data = None
        self.validation_data = None
        self.train_size = None
        
        self.current_train_epoch = self.validation_epoch = self.epoch_names = None
        self.train_epoch_cycler = None

    def setup(self, stage):
        # changes epochs
        adata = ad.read_h5ad(self.anndata_file)
        if "epoch_names" not in adata.uns:
            raise KeyError("AnnData missing 'epoch_names' in .uns")
        
        self.epoch_names = adata.uns['epoch_names']

        self.validation_epoch = self.epoch_names[0]

        layer_names = list(adata.layers)
        self.train_data = {}
        for name in tqdm(self.epoch_names, desc='Loading training data for epochs'):
            self.train_data[name] = self.get_data(adata, name, dhs_split='train', sample_split='train')
            if name == self.validation_epoch:
                self.validation_data = {
                    self.validation_epoch: self.get_data(
                        adata,
                        self.validation_epoch,
                        dhs_split='val',
                        sample_split='train'
                    )
                }
            for layer in layer_names:
                if layer.endswith(name):
                    del adata.layers[layer]
                    gc.collect()
        
        self.train_epoch_cycler = cycle(self.epoch_names)

        logger.info(f"Finished setup. Available epochs: {self.epoch_names}")

    def train_dataset(self, epoch=None):
        if epoch is None:
            epoch = self.current_train_epoch

        # Create new dataset
        data = self.train_data[epoch]
        train_dataset = SequenceEmbedDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.train_dataset_kwargs,
        )
        return train_dataset

    def validation_dataset(self):
        data = self.validation_data[self.validation_epoch]

        valid_dataset = SequenceEmbedDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.valid_dataset_kwargs,
        )
        return valid_dataset

    def train_dataloader(self):
        self.current_train_epoch = next(self.train_epoch_cycler)
        # Cycle to next file index
        # Create new dataloader
        data_loader = DataLoader(
            self.train_dataset(),
            shuffle=True,
            **self.dataloader_kwargs,
        )
        print('Finished creating train dataloader for epoch:', self.current_train_epoch, flush=True)
        return data_loader
    
    def teardown(self, stage: str):
        print("Teardown datamodule and free memory")
        print(stage)
        gc.collect()
    
    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset(),
            shuffle=False,
            **self.dataloader_kwargs,
        )

    @staticmethod
    def get_data(full_adata: ad.AnnData, suffix, dhs_split='train', sample_split='train'):
        """
        Generic method to extract data from anndata object for a given split

        Args:
            full_adata (ad.AnnData): Full AnnData object containing all data
            suffix (str): suffix of the epoch to extract. Gets added to layer names as `{layer}.{suffix}`
            dhs_split (str): which DHS split to use (train/val/test)
            sample_split (str): which sample split to use (train/val/test)

        Returns:
            data (dict): dictionary with extracted data
            embeddings_df (pd.DataFrame): DataFrame with extracted embeddings
        """
        adata = full_adata[
            full_adata.obsm['split_data'] == sample_split,
            full_adata.varm['split_data'] == dhs_split
        ]

        return extract_data_from_train_anndata(adata, suffix)

class SeqEmbedVariantDataModule(SeqEmbedDataModule):
    """
    Variant version of SeqEmbedDataModule.

    Differences:
      - Uses extract_var_data_from_anndata
      - Uses VariantEmbedDataset instead of SequenceEmbedDataset
    """
    def train_dataset(self, epoch=None):
        if epoch is None:
            epoch = self.current_train_epoch

        data = self.train_data[epoch]   # <-- already populated in setup()
        return VariantEmbedDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.train_dataset_kwargs,
        )

    def validation_dataset(self):
        data = self.validation_data[self.validation_epoch]  # <-- already populated in setup()
        return VariantEmbedDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.valid_dataset_kwargs,
        )


    @staticmethod
    def get_data(full_adata: ad.AnnData, suffix: str,
                 dhs_split="train", sample_split="train"):
        """Extract variant-level data from AnnData."""
        adata = full_adata[
            full_adata.obsm["split_data"] == sample_split,
            full_adata.varm["split_data"] == dhs_split,
        ]
        return extract_variant_data_from_anndata(adata, suffix)
