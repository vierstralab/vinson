import pytorch_lightning as L
from torch.utils.data import DataLoader
from vinson.datasets.sequence import SequenceEmbedDataset
from itertools import cycle, islice
from torchdata.stateful_dataloader import StatefulDataLoader

import anndata as ad
import numpy as np


class SeqEmbedDataModule(L.LightningDataModule):
    def __init__(
        self,
        adata: ad.AnnData,
        fasta_file,
        genotype_file=None,
        train_dataset_kwargs={},
        valid_dataset_kwargs={},
        dataloader_kwargs={},
        worker_init_fn=None
    ):
        super().__init__()
        self.worker_init_fn = worker_init_fn

        self.fasta_file = fasta_file
        self.adata = adata
        self.genotype_file = genotype_file

        self.train_dataset_kwargs = train_dataset_kwargs
        self.valid_dataset_kwargs = valid_dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        
        self.i = 0 # start with file 1
        
        self.train_dataset = None
        self.valid_dataset = None
        self.train_epoch_cycler = None
        # self.train_dl = None
    
    def setup(self, stage):
        # changes epochs
        n_epochs = self.adata.uns['n_epochs']
        self.train_epoch_cycler = islice(cycle(range(n_epochs)), self.i, None)

        # self.iterate_train_dataset()

    def train_dataloader(self):

        # Cycle to next file index
        self.i = next(self.train_epoch_cycler)
        data, embeddings_df = self.get_data(self.i, 'train', 'train')

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
            worker_init_fn=self.worker_init_fn,
        )
    

    def val_dataloader(self):
        data, embeddings_df = self.get_data(0, 'val', 'train')

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
            worker_init_fn=self.worker_init_fn,
        )

    def get_data(self, epoch, dhs_split, sample_split='train'):
        # FIX epoch
        adata_slice = self.adata[
            self.adata.obsm['split_data'] == sample_split,
            self.adata.varm['split_data'] == dhs_split
        ]
        class_coo = adata_slice.layers['class'].tocoo()
        row_idx = class_coo.row
        col_idx = class_coo.col

        data = {
            'density': adata_slice.layers['density'].tocoo().data,
            'sample_id': adata_slice.obs_names[row_idx],
            'background': adata_slice.layers['background'].tocoo().data,
            'read_depth': adata_slice.obs['nuclear_reads'].values[row_idx],
            'chrom': adata_slice.var['#chr'].values[col_idx],
            'summit': adata_slice.var['dhs_summit'].values[col_idx],
            'class': class_coo.data,
        }
        if 'indiv_id' in adata_slice.obsm:
            # maybe there is something more elegant
            indiv_ids = np.array(
                [x if x != "None" else None for x in adata_slice.obsm['indiv_id']]
            )
            data['indiv_id'] = indiv_ids[row_idx]
        
        embeddings_df = self.adata.obsm['motif_embeddings']
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
