import numpy as np

import h5py
import pandas as pd

from .encoding import sanitize_data


class VinsonData:
    """
    Class for handling Vinson data formatting and conversion.

    Parameters
    ----------
    data : dict
        Dictionary containing sample metadata. {
            'chrom': np.array, 'summit': np.array, etc.
        }
    encodings : dict
        Dictionary containing encodings for categorical variables. {
            'chrom': np.array, ...
        }
    embeddings_df : pd.DataFrame
        DataFrame of cell-type/state embeddings indexed by sample ID.
    """
    def __init__(self, data: dict, encodings: dict, embeddings_df: pd.DataFrame=None, is_variant=False):
        self.data = data
        self.encodings = encodings
        self.embeddings_df = embeddings_df
        self.is_variant = is_variant

        self.length = len(self.data['chrom'])
        for key, value in self.data.items():
            assert len(value) == self.length, f"All data arrays must have the same length. Key {key} has length {len(value)}, expected {self.length}."
    
    def __repr__(self):
        return f"VinsonData with keys: {list(self.data.keys())}. Encoded columns: {list(self.encodings.keys())}."
    
    def decode(self, key: str) -> np.ndarray:
        return self._decode(key, self.data[key])
    
    def _decode(self, key: str, indices: np.ndarray) -> np.ndarray:
        """
        Decode encoded values for a given key.
        
        Parameters:
            key (str): Key of the data dictionary to decode.
            indices (np.ndarray): Encoded indices to decode.
        """
        if key not in self.encodings:
            raise ValueError(f"Key {key} is not encoded.")
        return self.encodings[key][indices]

    def to_df(self) -> pd.DataFrame:
        """Convert data dictionary to pandas DataFrame."""
        data_df = pd.DataFrame(self.data)
        for key, enc in self.encodings.items():
            data_df[key] = pd.Categorical.from_codes(data_df[key], enc)
        return data_df
    
    def __len__(self):
        """Return number of samples in the dataset."""
        return self.length
    
    def keys(self):
        """Return keys of the data dictionary."""
        return self.data.keys()
    
    def __contains__(self, key):
        """Check if key is in the data dictionary."""
        return key in self.data.keys()

    def __getitem__(self, i) -> dict:
        """Get data dict for a given index."""
        return_dict = {}
        for key, value in self.data.items():
            return_dict[key] = value[i]
            if key in self.encodings:
                return_dict[key] = self._decode(key, return_dict[key])
                
        return return_dict

    def write_h5(self, h5_file: str):
        """Convert data dictionary to H5 file."""
        strings_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(h5_file, 'w') as f:
            for key, value in self.data.items():
                if key in self.encodings:
                    value = np.asarray(self.decode(key), dtype=strings_dtype)
                f.create_dataset(key, data=value, compression="gzip")

    @classmethod
    def from_raw(cls, raw_data: dict, embeddings_df: pd.DataFrame=None, is_variant=False):
        data, encodings = sanitize_data(raw_data, is_variant=is_variant)
        return cls(data, encodings, embeddings_df, is_variant=is_variant)
