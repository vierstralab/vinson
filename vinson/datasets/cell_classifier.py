import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder

class CellEncoderDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

        self.encoder = LabelEncoder()
        self.encoder.fit(self.labels)

        self.labels_encoded = self.encoder.transform(self.labels)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, i):
        return self.embeddings[i], self.labels_encoded[i]

    def num_classes(self):
        return len(self.encoder.classes_)