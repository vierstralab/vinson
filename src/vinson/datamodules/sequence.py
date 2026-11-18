import lightning.pytorch as L

from torch.utils.data import DataLoader
from vinson.utils.data_formatting import extract_data_from_train_anndata
from vinson.datasets.sequence import SequenceEmbedDataset, SequenceEmbedProfileDataset
from itertools import cycle

import anndata as ad
import gc

import pandas as pd

# TODO: move all logging to one helper file
import logging

from vinson.utils.data_formatting import VinsonData
from vinson.utils.helpers import read_h5ad_elem, read_h5ad_epoch

logger = logging.getLogger(__name__)


class SeqEmbedDataModule(L.LightningDataModule):
    def __init__(
        self,
        anndata_file: str,
        fasta_file: str,
        genotype_file = None,
        train_dataset_kwargs={},
        valid_dataset_kwargs={},
        pre_jitter=False,
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
        self.pre_jitter = pre_jitter
        self.train_data = None
        self.validation_data = None
        self.train_size = None

        self.current_train_epoch = self.validation_epoch = self.epoch_names = None
        self.train_epoch_cycler = None

    def setup(self, stage):
        # changes epochs
        self.epoch_names = read_h5ad_elem(self.anndata_file, "uns/epoch_names")

        self.validation_epoch = self.epoch_names[0]
        adata = read_h5ad_epoch(self.anndata_file, self.validation_epoch)
        self.validation_data = {
            self.validation_epoch: self.get_data(
                adata,
                self.validation_epoch,
                dhs_split="val",
                sample_split="train",
                pre_jitter=self.pre_jitter,
            )
        }
        # Only one training epoch is held in memory at a time, loaded in `train_dataloader`
        self.train_data = {}

        self.train_epoch_cycler = cycle(self.epoch_names)

        logger.info(f"Finished setup. Available epochs: {self.epoch_names}")

    def load_train_epoch(self, epoch):
        """
        Replace the training data held in memory with the data of `epoch`.
        """
        # free the previous epoch before reading the next one
        self.train_data = {}
        gc.collect()

        print(f"Loading training data for epoch {epoch}...", flush=True)
        adata = read_h5ad_epoch(self.anndata_file, epoch)
        self.train_data = {
            epoch: self.get_data(
                adata,
                epoch,
                dhs_split="train",
                sample_split="train",
                pre_jitter=self.pre_jitter
            )
        }
        del adata
        gc.collect()

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
        self.load_train_epoch(self.current_train_epoch)
        # Cycle to next file index
        # Create new dataloader
        data_loader = DataLoader(
            self.train_dataset(),
            shuffle=True,
            **self.dataloader_kwargs,
        )
        return data_loader

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset(),
            shuffle=False,
            **self.dataloader_kwargs,
        )

    @staticmethod
    def get_data(
        full_adata: ad.AnnData,
        suffix,
        dhs_split="train",
        sample_split="train",
        pre_jitter=False
    ) -> VinsonData:
        """
        Generic method to extract data from anndata object for a given split

        Args:
            full_adata (ad.AnnData): Full AnnData object containing all data
            suffix (str): suffix of the epoch to extract. Gets added to layer names as `{layer}.{suffix}`
            dhs_split (str): which DHS split to use (train/val/test)
            sample_split (str): which sample split to use (train/val/test)

        Returns:
            VinsonData: extracted data object
        """
        adata = full_adata[
            full_adata.obsm["split_data"] == sample_split,
            full_adata.varm["split_data"] == dhs_split,
        ]

        return extract_data_from_train_anndata(adata, suffix, pre_jitter=pre_jitter)


class SeqEmbedProfileDataModule(SeqEmbedDataModule):
    def __init__(self, anndata_file, fasta_file, dm_mu_params_file, dm_r_params_file, fp_figwig_file, **kwargs):
        super().__init__(anndata_file, fasta_file, **kwargs)
        self.dm_mu_params_file = dm_mu_params_file
        self.dm_r_params_file = dm_r_params_file
        self.fp_figwig_file = fp_figwig_file
        self.dm_mu_params = self.dm_r_params = None

    def setup(self, stage):
        self.dm_mu_params = pd.read_table(self.dm_mu_params_file, index_col=0)
        self.dm_r_params = pd.read_table(self.dm_r_params_file, index_col=0)
        super().setup(stage)

    def _dataset(self, data, dataset_kwargs):
        return SequenceEmbedProfileDataset(
            data=data,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            dm_mu_params=self.dm_mu_params,
            dm_r_params=self.dm_r_params,
            fp_figwig_file=self.fp_figwig_file,
            **dataset_kwargs,
        )

    def train_dataset(self, epoch=None):
        epoch = self.current_train_epoch if epoch is None else epoch
        return self._dataset(self.train_data[epoch], self.train_dataset_kwargs)

    def validation_dataset(self):
        return self._dataset(self.validation_data[self.validation_epoch], self.valid_dataset_kwargs)
