import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder


class EncoderDataset(Dataset):
    """ """

    def __init__(self, embeddings, labels, encoders={}, noise=0, seed=0):
        assert all(embeddings.index.isin(labels.index)), ""

        self.embeddings = embeddings
        self.labels = labels.loc[self.embeddings.index]

        self.encoders = encoders

        self.noise = noise
        self.seed = seed

        self.random_state = np.random.RandomState(self.seed)

        for col in self.labels.columns:
            if col not in self.encoders:
                _encoder = LabelEncoder()
                _encoder.fit(self.labels[col])
                self.encoders[col] = _encoder

        self._labels = self.labels.apply(lambda x: self.encoders[x.name].transform(x)).astype(int)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, i):
        """ """
        embed = self.embeddings.iloc[i].values.astype(np.float32)
        
        if self.noise > 0:
            embed  = embed  + self.random_state.normal(0, self.noise, len(embed)).astype(np.float32)

        x = {"embed": embed}
        y = self._labels.iloc[i].to_dict()
        return {**x, **y}
