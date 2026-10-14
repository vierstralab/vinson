import numpy as np

import h5py
import pandas as pd
from typing import List

from .sanitize import sanitize_data, ModelDataSchema, DHS, VARIANT
import logging

logger = logging.getLogger(__name__)

# Left old here in case we need it, united VinsonData is below

# class AbstractVinsonData:
#     """
#     Abstract class for handling Vinson data formatting and conversion. Handles encoding of categorical/string variables to heavily reduce cpu usage.

#     Parameters
#     ----------
#     data : dict
#         Dictionary containing sample metadata. {
#             'chrom': np.array, 'summit': np.array, etc.
#         }
#     encodings : dict
#         Dictionary containing encodings for categorical variables. {
#             'chrom': np.array, ...
#         }
#     embeddings_df : pd.DataFrame
#         DataFrame of cell-type/state embeddings indexed by sample ID.

#     Abstract class, use VinsonData or VinsonVariantData instead. Children should have a schema attribute
#     """
#     schema: ModelDataSchema = None


#     def __init__(self, data: dict, encodings: dict, embeddings_df: pd.DataFrame=None, *, is_inference):
#         self.data = data
#         self.encodings = encodings
#         self.embeddings_df = embeddings_df
#         self.is_inference = is_inference

#         self.length = len(self.data['chrom'])

#         if self.schema is None:
#             raise TypeError("Use VinsonData or VariantVinsonData.")
    
#         for key, value in self.data.items():
#             assert len(value) == self.length, f"All data arrays must have the same length. Key {key} has length {len(value)}, expected {self.length}."
    
#     def __repr__(self):
#         return f"{self.__class__.__name__}(n={self.length}, inference={self.is_inference}) with keys: {list(self.data.keys())}. Encoded columns: {list(self.encodings.keys())}."
    
#     def decode(self, key: str) -> np.ndarray:
#         return self._decode(key, self.data[key])
    
#     def _decode(self, key: str, indices: np.ndarray) -> np.ndarray:
#         """
#         Decode encoded values for a given key.
        
#         Parameters:
#             key (str): Key of the data dictionary to decode.
#             indices (np.ndarray): Encoded indices to decode.
#         """
#         if key not in self.encodings:
#             raise ValueError(f"Key {key} is not encoded.")
#         return self.encodings[key][indices]

#     def keys(self):
#         """Return keys of the data dictionary."""
#         return self.data.keys()
    
#     def __len__(self):
#         """Return number of samples in the dataset."""
#         return self.length

#     def __contains__(self, key):
#         """Check if key is in the data dictionary."""
#         return key in self.keys()

#     def _extract_normalized_value(self, key, x):
#         if key in self.encodings:
#             x = self._decode(key, x)

#             # scalar string
#             if isinstance(x, (np.str_, str)):
#                 x = str(x)
#             # vector of strings
#             elif isinstance(x, np.ndarray) and x.dtype == object:
#                 x = [str(t) for t in x.tolist()]
#             # vector of numpy unicode strings (if any remain)
#             elif isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S"):
#                 x = x.astype(str).tolist()
#         return x

#     def __getitem__(self, i) -> dict:
#         """Get data dict for a given index."""
#         return_dict = {}
#         for key, value in self.data.items():
#             x = value[i]

#             x = self._extract_normalized_value(key, x)

#             return_dict[key] = x
#         return return_dict

#     # writers
#     def to_df(self) -> pd.DataFrame:
#         """Convert data dictionary to pandas DataFrame."""
#         data_df = pd.DataFrame(self.data)
#         for key, enc in self.encodings.items():
#             data_df[key] = pd.Categorical.from_codes(data_df[key], enc)
#         return data_df

#     def to_dict(self) -> dict:
#         return {
#             key: self._extract_normalized_value(key, value) 
#                 for key, value in self.data.items()
#         }

#     def write_h5(self, h5_file: str):
#         """Convert data dictionary to H5 file."""
#         strings_dtype = h5py.string_dtype(encoding='utf-8')
#         with h5py.File(h5_file, 'w') as f:
#             for key, value in self.data.items():
#                 if key in self.encodings:
#                     value = np.asarray(self.decode(key), dtype=strings_dtype)
#                 f.create_dataset(key, data=value, compression="gzip")

#     # readers
#     @classmethod
#     def from_h5(cls, h5_file, embeddings_df: pd.DataFrame=None, *, is_inference=False):
#         with h5py.File(h5_file, 'r') as f:
#             data = {}
#             for key in f.keys():
#                 data[key] = f[key][()]

#         return cls.from_raw(data, embeddings_df, is_inference=is_inference)

#     @classmethod
#     def from_raw(cls, raw_data: dict, embeddings_df: pd.DataFrame=None, *, is_inference=False):
#         if cls.schema is None:
#             raise TypeError("No validation schema. Use VinsonData or VariantVinsonData.")
#         data, encodings = sanitize_data(cls.schema, raw_data, is_inference=is_inference)
#         return cls(data, encodings, embeddings_df, is_inference=is_inference)

#     @classmethod
#     def from_df(cls, df: pd.DataFrame, embeddings_df: pd.DataFrame = None, *, is_inference=False):
#         raw_data = {key: df[key].to_numpy() for key in df.columns}
#         return cls.from_raw(raw_data, embeddings_df, is_inference=is_inference)


# class VinsonData(AbstractVinsonData):
#     schema = DHS

#     @classmethod
#     def inference_from_bed(cls, bed_df: pd.DataFrame, sample_ids: List[str], embeddings_df: pd.DataFrame = None):
#         assert all(x in bed_df.columns for x in ['chrom', 'start', 'end'])
#         df = bed_df.copy()
#         if 'summit' not in df.columns:
#             df['summit'] = df.eval('(start + end) // 2')
#         sample_ids = list(sample_ids)
#         df['sample_id'] = [sample_ids for _ in range(len(df))]
#         df = df.explode(['sample_id'])
#         return cls.from_df(df, embeddings_df=embeddings_df, is_inference=True)


# class VariantVinsonData(AbstractVinsonData):
#     schema = VARIANT



class VinsonData:
    """
    Abstract class for handling Vinson data formatting and conversion. Handles encoding of categorical/string variables to heavily reduce cpu usage.

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

    Abstract class, use VinsonData or VinsonVariantData instead. Children should have a schema attribute
    """
    
    schema: ModelDataSchema = None

    def __init__(self, data: dict, encodings: dict, embeddings_df: pd.DataFrame=None,*, is_inference):
        self.data = data
        self.encodings = encodings
        self.embeddings_df = embeddings_df
        self.is_inference = is_inference
        self.length = len(self.data['chrom'])

        if self.schema is None:
            raise TypeError("Use DHS or Variant as schema.")
    
        for key, value in self.data.items():
            assert len(value) == self.length, f"All data arrays must have the same length. Key {key} has length {len(value)}, expected {self.length}."
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.length}, inference={self.is_inference}) with keys: {list(self.data.keys())}. Encoded columns: {list(self.encodings.keys())}."
    
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

    def keys(self):
        """Return keys of the data dictionary."""
        return self.data.keys()
    
    def __len__(self):
        """Return number of samples in the dataset."""
        return self.length

    def __contains__(self, key):
        """Check if key is in the data dictionary."""
        return key in self.keys()

    def _extract_normalized_value(self, key, x):
        if key in self.encodings:
            x = self._decode(key, x)

            # scalar string
            if isinstance(x, (np.str_, str)):
                x = str(x)
            # vector of strings
            elif isinstance(x, np.ndarray) and x.dtype == object:
                x = [str(t) for t in x.tolist()]
            # vector of numpy unicode strings (if any remain)
            elif isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S"):
                x = x.astype(str).tolist()
        return x

    def __getitem__(self, i) -> dict:
        """Get data dict for a given index."""
        return_dict = {}
        for key, value in self.data.items():
            x = value[i]

            x = self._extract_normalized_value(key, x)

            return_dict[key] = x
        return return_dict

    # writers
    def to_df(self) -> pd.DataFrame:
        """Convert data dictionary to pandas DataFrame."""
        data_df = pd.DataFrame(self.data)
        for key, enc in self.encodings.items():
            data_df[key] = pd.Categorical.from_codes(data_df[key], enc)
        return data_df

    def to_dict(self) -> dict:
        return {
            key: self._extract_normalized_value(key, value) 
                for key, value in self.data.items()
        }

    def write_h5(self, h5_file: str):
        """Convert data dictionary to H5 file."""
        strings_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(h5_file, 'w') as f:
            for key, value in self.data.items():
                if key in self.encodings:
                    value = np.asarray(self.decode(key), dtype=strings_dtype)
                f.create_dataset(key, data=value, compression="gzip")

    # readers
    @classmethod
    def from_h5(cls, h5_file, embeddings_df: pd.DataFrame=None, *, is_inference=False):
        with h5py.File(h5_file, 'r') as f:
            data = {}
            for key in f.keys():
                data[key] = f[key][()]

        return cls.from_raw(data, embeddings_df, is_inference=is_inference)

    @classmethod
    def from_raw(cls, raw_data: dict, embeddings_df: pd.DataFrame=None, *, is_inference=False, is_variant:bool = False):
        ###conflict with genome-tools/plotting/modular_plot/api.py
        # if cls.schema is None or is_variant is None:                                         
        #     raise TypeError("No validation schema. Set is_variant argument as bool.")
        if cls.schema is None:
            if is_variant:
                cls.schema = VARIANT 
            elif not is_variant:
                cls.schema = DHS
        ###conflict with genome-tools/plotting/modular_plot/api.py

        data, encodings = sanitize_data(cls.schema, raw_data, is_inference=is_inference)
        return cls(data, encodings, embeddings_df, is_inference=is_inference)

    @classmethod
    def from_df(cls, df: pd.DataFrame, embeddings_df: pd.DataFrame = None, *, is_inference=False):
        raw_data = {key: df[key].to_numpy() for key in df.columns}
        return cls.from_raw(raw_data, embeddings_df, is_inference=is_inference)
    
    # logic with product [shifts x coords] is here, these two methods resemble each other a lot, not sure they need to be merged
    @classmethod
    def explode(cls, df, embeddings_meta_df = None, sample_ids = None, shifts=[0], revcomp = True):

        sample_ids = list(sample_ids)
        df['sample_id'] = [sample_ids for _ in range(len(df))]
        df['shift'] = [shifts for _ in range(len(df))]
        
        df = df.explode(['sample_id'])
    
        if embeddings_meta_df is not None:
            indiv_id_meta = embeddings_meta_df.loc[sample_ids, 'indiv_id']
            df = df.merge(indiv_id_meta, left_on='sample_id', right_index=True)            
            
        df = df.explode(['shift'])

        if revcomp:
            df['revcomp'] = [[True, False] for _ in range(len(df))]
            df = df.explode(['revcomp'])

        return df
    
    @classmethod
    def inference_from_bed(cls, bed_df: pd.DataFrame, 
                           embeddings_df: pd.DataFrame|None = None, embeddings_meta_df: pd.DataFrame|None = None,
                        sample_ids: List[str]= None, shifts: List[int] = [0], revcomp:bool=True):
        assert all(x in bed_df.columns for x in ['chrom', 'start', 'end'])
        cls.schema = DHS

        df = bed_df.copy()
        if 'summit' not in df.columns:
            df['summit'] = (df['start'] + df['end'] )// 2

        cartesian_df = cls.explode(df, embeddings_meta_df, sample_ids, shifts, revcomp)
        
        return cls.from_df(cartesian_df, embeddings_df=embeddings_df, is_inference=True)

    @classmethod
    def inference_from_variant(cls, variant_df: pd.DataFrame, 
                               embeddings_df: pd.DataFrame = None, embeddings_meta_df: pd.DataFrame = None,
                               sample_ids: List[str]= None, shifts: List[int] = [0], revcomp:bool = True): 
        assert all(x in variant_df.columns for x in ['chrom', 'pos', 'ref', 'alt', 'variant_id'])
        cls.schema = VARIANT

        df = variant_df.copy()

        cartesian_df = cls.explode(df, embeddings_meta_df, sample_ids, shifts, revcomp)
                
        return cls.from_df(cartesian_df, embeddings_df=embeddings_df, is_inference=True)



