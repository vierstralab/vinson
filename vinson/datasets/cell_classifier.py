import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder


class EncoderDataset(Dataset):
    """ """
    def __init__(self, embeddings, labels, encoders={}):
        
        assert all(embeddings.index.isin(labels.index)), ""
        
        self.embedding = embeddings
        self.labels = labels.drop_duplicates().loc[self.embeddings.index]



        self.encoders = encoders

        for col in self.labels.columns:
            if col not in self.encoders:
                _encoder = LabelEncoder()
                _encoder.fit(self.labels_df[col])
                self.encoders[col] = _encoder

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, i):
        ret = {"embed": self.embeddings.iloc[i]}

        encoded_labels = {
            k: self.encoders[k].transform(v) for k, v in self.labels.iloc[i].items()
        }

        return ret.update(encoded_labels)
