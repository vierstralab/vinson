import numpy as np

import h5py
import pandas as pd
from typing import List, Iterable

from .sanitize import sanitize_data, ModelDataSchema, DHS, VARIANT
import logging

logger = logging.getLogger(__name__)



class VinsonData:
    """
    Container holding one row per training/prediction example (DHS/accessibility data), validated
    against `schema` (see `sanitize.py`). Handles encoding of categorical/string variables to heavily
    reduce memory/cpu usage.

    Parameters
    ----------
    data : dict
        One array per field (e.g. 'chrom', 'summit', 'sample_id', 'class', 'density', ...), all the
        same length. Categorical fields are stored as integer codes; see `encodings`.
    encodings : dict
        Maps each encoded field name to the array of original string values, e.g. {'chrom': np.array([...])}.
        `data[key][i]` indexes into `encodings[key]` to recover the original value at row `i` (see `decode`).
    embeddings_df : pd.DataFrame, optional
        Per-sample cell-type/state embeddings, indexed by `sample_id`.
    is_inference : bool, keyword-only
        If False (training/validation), `schema`'s target fields (e.g. `class`/`density` here,
        `ref_counts`/`total_counts`/... for `VariantVinsonData`) must be present, and only
        training datasets (e.g. `SequenceEmbedDataset`) will accept the container. If True
        (prediction-only), target fields are optional, and only inference datasets (e.g.
        `InferenceSequenceEmbedDataset`) will accept it.

    Use `VinsonData` for DHS/accessibility data and the sibling `VariantVinsonData` for variant-effect
    data; each declares its own `schema`. Prefer building a container via `from_h5`, `from_raw`, `from_df`,
    `inference_from_bed` (`inference_from_variant` for `VariantVinsonData`), or one of the module-level
    `extract_data_from_*` helpers in `readers.py`, rather than calling `__init__` directly.
    """
    
    schema: ModelDataSchema = DHS

    def __init__(self, data: dict, encodings: dict, embeddings_df: pd.DataFrame=None,*, is_inference):
        self.data = data
        self.encodings = encodings
        self.embeddings_df = embeddings_df
        self.is_inference = is_inference
        self.length = len(self.data['chrom'])
    
        for key, value in self.data.items():
            assert len(value) == self.length, f"All data arrays must have the same length. Key {key} has length {len(value)}, expected {self.length}."
    
    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.length}, inference={self.is_inference}) with keys: {list(self.data.keys())}. Encoded columns: {list(self.encodings.keys())}."
    
    def decode(self, key: str) -> np.ndarray:
        """Decode an encoded field back to its original (string) values for every row."""
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
        """Return a dict of decoded (human-readable) field arrays, one entry per field."""
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
        """
        Load a container from an `.h5` file previously written by `write_h5`, pairing it with
        `embeddings_df` (typically `ref_adata.obsm['motif_embeddings']`).
        """
        with h5py.File(h5_file, 'r') as f:
            data = {}
            for key in f.keys():
                data[key] = f[key][()]

        return cls.from_raw(data, embeddings_df, is_inference=is_inference)

    @classmethod
    def from_raw(cls, raw_data: dict, embeddings_df: pd.DataFrame=None, *, is_inference=False):
        """Build a container from raw (un-encoded) field arrays, validating and encoding them against `cls.schema`."""
        data, encodings = sanitize_data(cls.schema, raw_data, is_inference=is_inference)
        return cls(data, encodings, embeddings_df, is_inference=is_inference)

    @classmethod
    def from_df(cls, df: pd.DataFrame, embeddings_df: pd.DataFrame = None, *, is_inference=False):
        """Build a container from a DataFrame with one column per field (e.g. `some_container.to_df()`)."""
        raw_data = {key: df[key].to_numpy() for key in df.columns}
        return cls.from_raw(raw_data, embeddings_df, is_inference=is_inference)

    # logic with product [shifts x coords] is here, these two methods resemble each other a lot, not sure they need to be merged
    @staticmethod
    def explode(df, sample_indiv_mapping=None, sample_ids=None, shifts=(0, ), revcomp=False):
        """
        Cartesian-expand `df` (one row per region/variant) into one row per
        `(region/variant, sample_id, shift[, revcomp])` combination, for building an inference
        container that scores every combination explicitly instead of relying on random augmentation.

        Parameters
        ----------
        df : pd.DataFrame
            One row per region/variant to score.
        sample_indiv_mapping : pd.DataFrame, optional
            `indiv_id` column indexed by `sample_id`; if given, each exploded row is also merged with
            the `indiv_id` of its sample (for genotype injection).
        sample_ids : list[str]
            Samples to score every row against.
        shifts : Iterable[int], default (0,)
            Genomic-window shifts (bp) to score every row at, for test-time augmentation.
        revcomp : bool, default False
            If True, also explode into the forward and reverse-complement strand for every row.

        Used internally by `inference_from_bed` / `inference_from_variant`.
        """
        sample_ids = list(sample_ids)
        df['sample_id'] = [sample_ids for _ in range(len(df))]
        df['shift'] = [shifts for _ in range(len(df))]

        df = df.explode(['sample_id'])

        if sample_indiv_mapping is not None:
            indiv_id_meta = sample_indiv_mapping.loc[sample_ids, 'indiv_id']
            df = df.merge(indiv_id_meta, left_on='sample_id', right_index=True)

        df = df.explode(['shift'])

        if revcomp:
            df['revcomp'] = [[True, False] for _ in range(len(df))]
            df = df.explode(['revcomp'])

        return df

    @classmethod
    def inference_from_bed(
        cls,
        bed_df: pd.DataFrame,
        embeddings_df: pd.DataFrame = None,
        sample_indiv_mapping: pd.DataFrame = None,
        sample_ids: List[str]= None,
        shifts: Iterable[int] = (0,),
        revcomp: bool=False
    ):
        """
        Build an inference-only (`is_inference=True`) container from a BED-like dataframe
        (`chrom`/`start`/`end`, optionally `summit`), expanding every region into one row per
        `(region, sample_id, shift[, revcomp])` combination (see `explode`) for scoring with
        `InferenceSequenceOnlyDataset` / `InferenceSequenceEmbedDataset`.

        Parameters
        ----------
        bed_df : pd.DataFrame
            Regions to score; must have `chrom`, `start`, `end` (and optionally `summit`, computed as
            the midpoint of `start`/`end` if absent).
        embeddings_df : pd.DataFrame, optional
            Per-sample cell-type/state embeddings, indexed by `sample_id`.
        sample_indiv_mapping : pd.DataFrame, optional
            `indiv_id` column indexed by `sample_id`, for genotype injection (requires a `genotype_file`
            at dataset construction).
        sample_ids : list[str]
            Samples to score every region against.
        shifts : Iterable[int], default (0,)
            Genomic-window shifts (bp) to score every region at, for test-time augmentation.
        revcomp : bool, default False
            If True, score both strands for every region.
        """
        df = bed_df.copy()
        if 'summit' not in df.columns:
            df['summit'] = (df['start'] + df['end'] ) // 2

        cartesian_df = cls.explode(df, sample_indiv_mapping, sample_ids, shifts, revcomp)

        return cls.from_df(cartesian_df, embeddings_df=embeddings_df, is_inference=True)


class VariantVinsonData(VinsonData):
    """
    Variant-effect counterpart of `VinsonData` — same container interface, but validated against the
    `VARIANT` schema (`chrom`, `pos`, `ref`, `alt`, `sample_id`, targets `ref_counts`/`total_counts`/
    `BAD`/`logit_es`). Paired with `VariantEmbedDataset` (training) / `InferenceVariantEmbedDataset`
    (inference) in `vinson/datasets/variant.py`.
    """

    schema: ModelDataSchema = VARIANT

    @classmethod
    def inference_from_variant(
        cls,
        variant_df: pd.DataFrame,
        sample_ids: List[str],
        embeddings_df: pd.DataFrame = None,
        sample_indiv_mapping: pd.DataFrame = None,
        shifts: Iterable[int] = (0,),
        revcomp: bool = True
    ):
        """
        Build an inference-only (`is_inference=True`) container from a variant dataframe
        (`chrom`/`pos`/`ref`/`alt`), expanding every variant into one row per
        `(variant, sample_id, shift[, revcomp])` combination (see `explode`) for scoring with
        `InferenceVariantEmbedDataset`.

        Parameters
        ----------
        variant_df : pd.DataFrame
            Variants to score; must have `chrom`, `pos`, `ref`, `alt`.
        embeddings_df : pd.DataFrame, optional
            Per-sample cell-type/state embeddings, indexed by `sample_id`.
        sample_indiv_mapping : pd.DataFrame, optional
            `indiv_id` column indexed by `sample_id`, for genotype injection (requires a `genotype_file`
            at dataset construction).
        sample_ids : list[str]
            Samples to score every variant against.
        shifts : Iterable[int], default (0,)
            Genomic-window shifts (bp) to score every variant at, for test-time augmentation.
        revcomp : bool, default True
            If True (the default here, unlike `inference_from_bed`), score both strands for every variant.
        """
        df = variant_df.copy()

        cartesian_df = cls.explode(df, sample_indiv_mapping, sample_ids, shifts, revcomp)

        return cls.from_df(cartesian_df, embeddings_df=embeddings_df, is_inference=True)

