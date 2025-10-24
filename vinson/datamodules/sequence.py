import lightning.pytorch as L
from torch.utils.data import DataLoader
from vinson.datasets.sequence import SequenceEmbedDataset
from itertools import cycle
from torchdata.stateful_dataloader import StatefulDataLoader

import anndata as ad
import numpy as np

# move all logging to one helper file
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
        dataloader_kwargs={},
    ):
        """
        Initialize the SeqEmbedDataModule.

        Args:
            adata (ad.AnnData): AnnData object containing the dataset.
            fasta_file (str): Path to the FASTA file.
            genotype_file (str, optional): Path to the genotype file.
            train_dataset_kwargs (dict, optional): Additional arguments for the training dataset.
            valid_dataset_kwargs (dict, optional): Additional arguments for the validation dataset.
            dataloader_kwargs (dict, optional): Additional arguments for train and validation dataloaders.
        """
        super().__init__()

        self.fasta_file = fasta_file
        self.anndata_file = anndata_file

        self.genotype_file = genotype_file

        self.train_dataset_kwargs = train_dataset_kwargs
        self.valid_dataset_kwargs = valid_dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        
        self.adata = None
        self.current_train_epoch = self.validation_epoch = self.epoch_names = None
        self.train_dataset = self.valid_dataset = self.train_epoch_cycler = None
        # self.train_dl = None

    def setup(self, stage):
        # changes epochs
        self.adata = ad.read_h5ad(self.anndata_file)
        self.epoch_names = self.adata.uns['epoch_names']
        self.train_epoch_cycler = cycle(self.epoch_names)
        self.validation_epoch = self.epoch_names[0]
        logger.info(f"Finished setup. Available epochs: {self.epoch_names}")

    # def read_adata(self):
    #     return ad.read_zarr(self.anndata_file)

    def train_dataloader(self):
        # Cycle to next file index
        self.current_train_epoch = next(self.train_epoch_cycler)
        data, embeddings_df = self.get_data(self.current_train_epoch, 'train', 'train')

        # Create new dataset
        self.train_dataset = SequenceEmbedDataset(
            data=data,
            embeddings_df=embeddings_df,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.train_dataset_kwargs,
        )
        # Create new dataloader
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.dataloader_kwargs,
        )
    
    def val_dataloader(self):
        data, embeddings_df = self.get_data(self.validation_epoch, 'val', 'train')

        self.valid_dataset = SequenceEmbedDataset(
            data=data,
            embeddings_df=embeddings_df,
            fasta_file=self.fasta_file,
            genotype_file=self.genotype_file,
            **self.valid_dataset_kwargs,
        )
        # Create new dataloader
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def get_data(self, name, dhs_split='train', sample_split='train'):
        """
        Generic method to extract data from anndata object for a given split

        Args:
            name (str): name of the epoch to extract. Gets added to layer names as `{layer}.{name}`
            dhs_split (str): which DHS split to use (train/val/test)
            sample_split (str): which sample split to use (train/val/test)

        Returns:
            data (dict): dictionary with extracted data
            embeddings_df (pd.DataFrame): DataFrame with extracted embeddings
        """
        full_adata = self.adata #self.read_adata()
        adata = full_adata[
            full_adata.obsm['split_data'] == sample_split,
            full_adata.varm['split_data'] == dhs_split
        ]
        print(f"Extracting data for {name}, dhs_split: {dhs_split}, sample_split: {sample_split}", flush=True)

        layers = {"class": None, "density": None, "mean_bg_agg_cutcounts": None}
        for layer_name in layers:
            epoch_layer_name = f"{layer_name}.{name}"
            layers[layer_name] = adata.layers[epoch_layer_name].tocoo()
            # for epoch_name in self.epoch_names:
            #    del full_adata.layers[f"{layer_name}.{epoch_name}"]

        print(f"Finished Extracting data for {name}, dhs_split: {dhs_split}, sample_split: {sample_split}", flush=True)

        class_coo = layers["class"]
        row_idx, col_idx = class_coo.row, class_coo.col

        data = {
            'read_depth': adata.obs['nuclear_reads'].values[row_idx],
            'sample_id': adata.obs_names[row_idx],
            'chrom': adata.var['#chr'].values[col_idx],
            'summit': adata.var['dhs_summit'].values[col_idx],
            'background': layers['mean_bg_agg_cutcounts'].data,
            'class': layers['class'].data,
            'density': layers['density'].data,
        }


        if 'indiv_id' in adata.obsm:
            logger.debug(
                f"Adding indiv_id from adata.obsm['indiv_id'] for {name}, dhs_split: {dhs_split}, sample_split: {sample_split}"
            )
            # maybe come up with something more elegant
            indiv_ids = np.array(
                [
                    x if x != "None" else None
                    for x in adata.obsm['indiv_id']
                ]
            )
            data['indiv_id'] = indiv_ids[row_idx]

        if 'dhs_weight' in adata.varm:
            logger.debug(
                f"Adding dhs_weight from adata.var['dhs_weight'] for {name}, dhs_split: {dhs_split}, sample_split: {sample_split}"
            )
            data['dhs_weight'] = adata.varm['dhs_weight'].values[col_idx]

        logger.info(
            f"Finished extracting data for {name}, dhs_split: {dhs_split}, sample_split: {sample_split}"
        )

        embeddings_df = adata.obsm['motif_embeddings']

        return data, embeddings_df

    # def iterate_train_dataset(self):
    #     """ """
    #     # Cycle to next file index
    #     self.i = next(self.train_file_cycler)
        
    #     # Create new dataset
    #     self.train_dataset = SequenceEmbedDataset(
    #         self.train_samples_files[self.i],
    #         self.embeddings_file,
    #         self.fasta_file,
    #         negative_samples_file=self.train_samples_negative_files[self.i],
    #         **self.train_dataset_kwargs,
    #     )

    #     self.train_dl = StatefulDataLoader(
    #         self.train_dataset,
    #         shuffle=True,
    #         **self.dataloader_kwargs,
    #         worker_init_fn=self.worker_init_fn,
    #     )

    # def train_dataloader(self):
    #     return self.train_dl
    
    # def state_dict(self):
    #     state = {
    #         "embeddings_file": self.embeddings_file,
    #         "fasta_file": self.fasta_file,
    #         "train_samples_files": self.train_samples_files,
    #         "train_samples_negative_files": self.train_samples_negative_files,
    #         "valid_samples_file": self.valid_samples_file,
    #         "valid_samples_negative_file": self.valid_samples_negative_file,
    #         "train_dataset_kwargs": self.train_dataset_kwargs,
    #         "valid_dataset_kwargs": self.valid_dataset_kwargs,
    #         "dataloader_kwargs": self.dataloader_kwargs,
    #         "i": self.i, # which file are currently using (epoch)
    #     }
    #     # state["train_dataloader"] = self.train_dataloader.state_dict()

    #     return state

    # def load_state_dict(self, state_dict):
    #     # Update module attributes
    #     # dl_state_dict = state_dict.pop("train_dataloader")
    #     # self.train_dataloader.load_state_dict(dl_state_dict)

    #     self.__dict__.update(state_dict)
