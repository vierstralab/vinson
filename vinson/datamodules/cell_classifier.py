import pytorch_lightning as L
from torch.utils.data import DataLoader
from vinson.datasets.sequence import SequenceEmbedDataset
from itertools import cycle, islice
from torchdata.stateful_dataloader import StatefulDataLoader


class SeqEmbedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_samples_files,
        train_samples_negative_files,
        valid_samples_file,
        valid_samples_negative_file,
        embeddings_file,
        fasta_file,
        train_dataset_kwargs={},
        valid_dataset_kwargs={},
        dataloader_kwargs={},
        worker_init_fn=None
    ):
        super().__init__()
        self.worker_init_fn = worker_init_fn

        self.embeddings_file = embeddings_file
        self.fasta_file = fasta_file

        assert len(train_samples_files) == len(train_samples_negative_files), (
            "Train samples and negative samples files must have same length!"
        )

        self.train_samples_files = sorted(train_samples_files)
        self.train_samples_negative_files = sorted(train_samples_negative_files)

        self.valid_samples_file = valid_samples_file
        self.valid_samples_negative_file = valid_samples_negative_file

        self.train_dataset_kwargs = train_dataset_kwargs
        self.valid_dataset_kwargs = valid_dataset_kwargs
        self.dataloader_kwargs = dataloader_kwargs
        
        self.i = 0 # start with file 1
        
        self.train_dataset = None
        self.valid_dataset = None
        self.train_file_cycler = None
        self.train_dl = None

    def setup(self, stage):
        # Set file cycler
        n_files = len(self.train_samples_files)
        self.train_file_cycler = islice(cycle(range(n_files)), self.i, None)

        # self.iterate_train_dataset()

    def iterate_train_dataset(self):
        """ """
        # Cycle to next file index
        self.i = next(self.train_file_cycler)
        
        # Create new dataset
        self.train_dataset = SequenceEmbedDataset(
            self.train_samples_files[self.i],
            self.embeddings_file,
            self.fasta_file,
            negative_samples_file=self.train_samples_negative_files[self.i],
            **self.train_dataset_kwargs,
        )

        self.train_dl = StatefulDataLoader(
            self.train_dataset,
            shuffle=True,
            **self.dataloader_kwargs,
            worker_init_fn=self.worker_init_fn,
        )

    # def train_dataloader(self):
    #     return self.train_dl

    def train_dataloader(self):

        # Cycle to next file index
        self.i = next(self.train_file_cycler)

        # Create new dataset
        self.train_dataset = SequenceEmbedDataset(
            self.train_samples_files[self.i],
            self.embeddings_file,
            self.fasta_file,
            negative_samples_file=self.train_samples_negative_files[self.i],
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
        self.valid_dataset = SequenceEmbedDataset(
            self.valid_samples_file,
            self.embeddings_file,
            self.fasta_file,
            negative_samples_file=self.valid_samples_negative_file,
            **self.valid_dataset_kwargs,
        )
        # Create new dataloader
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            **self.dataloader_kwargs,
            worker_init_fn=self.worker_init_fn,
        )
    
    def state_dict(self):
        state = {
            "embeddings_file": self.embeddings_file,
            "fasta_file": self.fasta_file,
            "train_samples_files": self.train_samples_files,
            "train_samples_negative_files": self.train_samples_negative_files,
            "valid_samples_file": self.valid_samples_file,
            "valid_samples_negative_file": self.valid_samples_negative_file,
            "train_dataset_kwargs": self.train_dataset_kwargs,
            "valid_dataset_kwargs": self.valid_dataset_kwargs,
            "dataloader_kwargs": self.dataloader_kwargs,
            "i": self.i, # which file are currently using (epoch)
        }
        # state["train_dataloader"] = self.train_dataloader.state_dict()

        return state

    def load_state_dict(self, state_dict):
        # Update module attributes
        # dl_state_dict = state_dict.pop("train_dataloader")
        # self.train_dataloader.load_state_dict(dl_state_dict)

        self.__dict__.update(state_dict)