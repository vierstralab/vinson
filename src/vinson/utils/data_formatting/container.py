from os.path import exists
import numpy as np

import h5py
import pandas as pd

from .encoding import sanitize_data
import logging

logger = logging.getLogger(__name__)

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
            # return_dict[key] = value[i]
            # if key in self.encodings:
            #     return_dict[key] = self._decode(key, return_dict[key])
            x = value[i]

            if key in self.encodings:
                # return_dict[key] = self._decode(key, return_dict[key])
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

            return_dict[key] = x
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



# don't want to store all combinations in dict, so made their assembly in getitem
# feel free to delete/refactor, just wanted to help =)

class CartesianVinsonData:
    """
    Class for handling Vinson data formatting and conversion.

    Parameters
    ----------
    data : dict
        Dictionary containing sample metadata. {
            'chrom': np.array, 'summit': np.array, etc.
        }
    embeddings_df : pd.DataFrame
        DataFrame of cell-type/state embeddings indexed by sample ID.
    encodings : dict
            Dictionary containing encodings for categorical variables. {
                'chrom': np.array, ...
            }
    """
    def __init__(self, data: dict, 
                 embeddings_df: pd.DataFrame,
                 fasta_file: str = None,
                 shifts: list= [0], 
                 embeddings_meta_df:pd.DataFrame = None, 
                 genotype_file:str|pd.DataFrame = None,
                 encodings:dict = {}):
        self.data = data
        self.embeds_ids = embeddings_df.index.values
        self.embeds_vals = embeddings_df.astype(np.float32).values
        self.embeddings_df = embeddings_df
        self.fasta_file = fasta_file
        self.shifts = np.asarray(shifts)

        logger.info(f'Number of samples that will be scored is {len(self.embeds_ids)}')

        self.embeddings_meta_df = embeddings_meta_df
        self.genotype_file = genotype_file
        
        is_genotype = genotype_file is not None and exists(genotype_file)
        is_embed_meta = embeddings_meta_df is not None 
        self.include_genotypes = is_embed_meta and is_genotype
        
        if self.include_genotypes:

            assert "indiv_id" in embeddings_meta_df.columns, (
                "embeddings_meta_df must include 'indiv_id' column.")
            assert np.isin(self.embeds_ids, embeddings_meta_df.index.values).all(), 'embeddings_meta_df index must include indexes from embeddings_df.'
            self.embeddings_meta_df = embeddings_meta_df.loc[embeddings_df.index.values].copy()
            self.genotype_file = genotype_file
            self.indiv_ids = self.embeddings_meta_df['indiv_id'].values
            
        else:

            logger.info(
                "No genotyping files provided -- continuing without sample genotypes.")
            self.embeddings_meta_df
        
        self.check()

        self.encodings = encodings
        self.shape = (self.embeddings_df.shape[0] , len(shifts) , len(self.data['chrom']))
        self.length = np.prod(self.shape)#self.embeddings_df.shape[0] * len(shifts) * len(self.data['chrom']) 
        logger.info(f'Total number of combinations that will be scored is {self.embeddings_df}')

    def check(self):
        bed_columns_se = ['chrom', 'start', 'end'] 
        bed_flag = all([
            col in self.data for col in bed_columns_se
        ])

        variant_columns = ['chrom', 'pos' ,'ref', 'alt', 'variant_id']
        var_flag = all([
            col in self.data.keys() for col in variant_columns
        ])

        assert bed_flag or var_flag, \
                    'You must provide either \n \
                    bed with the following columns [chrom, start, end]\n \
                    or table with [chrom, pos, ref, alt, variant_id]'
        
        length = len(self.data['chrom'])
        for key, value in self.data.items():
                    assert len(value) == length, f"All data arrays must have the same length. Key {key} has length {len(value)}, expected {length}."
                
        if (
            (self.embeddings_meta_df is not None) ^ (self.genotype_file is not None)
            ):
            raise ValueError('Both embeddings_meta_df and genotype_file should be passed to inject genotypes')
        
    def __repr__(self):
        return f"CartesianVinsonData with keys: {list(self.data.keys())}. Shape: {list(self.shape)}"
    
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
        x,y,z = np.unravel_index(i, self.shape)
        
        row = {}

        for key, value in self.data.items():
            val2ret = value[z]
            row[key] = val2ret
        
        row['embed'] = self.embeds_vals[x]
        row['shift'] = self.shifts[y]

        if self.include_genotypes:
            row['indiv_id'] = self.indiv_ids[x]

        return row

    def write_h5(self, h5_file: str):
        """Convert data dictionary to H5 file."""
        strings_dtype = h5py.string_dtype(encoding='utf-8')

        idxs = np.arange(len(self))
        embeds_idxs, shift_idxs, coord_idxs = np.unravel_index(idxs, self.shape)
        if len(self.shifts) > 1:
            self.data['shifts'] = self.shifts[shift_idxs] 
        self.data['embed_id'] = self.embeds_ids[embeds_idxs]
        
        with h5py.File(h5_file, 'w') as f:
            for key, value in self.data.items():
                if key in self.encodings:
                    value = np.asarray(self.decode(key), dtype=strings_dtype)
                f.create_dataset(key, data=value[coord_idxs], compression="gzip")

    @classmethod
    def from_raw(cls, raw_data: dict, embeddings_df: pd.DataFrame):
        data, encodings = sanitize_data(raw_data)
        return cls(data, embeddings_df, encodings=encodings)
    
    @classmethod
    def from_df(cls, coords_df:pd.DataFrame = None, embeddings_df:pd.DataFrame = None, 
                fasta_file:str = None, shifts:list=[0],
                embeddings_meta_df:pd.DataFrame = None, genotype_file:str|pd.DataFrame = None):
        data_dict = coords_df.to_dict(orient='list')
        return cls(data_dict, embeddings_df,fasta_file,shifts, embeddings_meta_df, genotype_file)
