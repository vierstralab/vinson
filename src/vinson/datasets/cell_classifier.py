import numpy as np

import torch
import pandas as pd
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder


class EncoderDataset(Dataset):
    """ """

    def __init__(self, embeddings: pd.DataFrame, labels: pd.DataFrame=None, encoders=None, noise=0, seed=0):
        if encoders is None:
            encoders = {}

        self.embeddings = embeddings
        
        self.encoders = encoders

        self.noise = noise
        self.seed = seed

        self.random_state = np.random.RandomState(self.seed)

        if labels is not None:
            self.labels = labels.loc[self.embeddings.index]
            for col in self.labels.columns:
                if col not in self.encoders:
                    _encoder = LabelEncoder()
                    _encoder.fit(self.labels[col])
                    self.encoders[col] = _encoder
            
            self._labels = self.labels.apply(
                lambda x: self.encoders[x.name].transform(x)
            ).astype(int)
        else:
            self.labels = None
            self._labels = None


    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, i):
        """ """
        embed = self.embeddings.iloc[i].values.astype(np.float32)
        
        if self.noise > 0:
            embed += self.random_state.normal(0, self.noise, len(embed)).astype(np.float32)

        x = {"embed": embed}
        if self._labels is None:
            return x
        y = self._labels.iloc[i].to_dict()
        return {**x, **y}
