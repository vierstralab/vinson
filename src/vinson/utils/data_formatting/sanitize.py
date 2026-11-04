import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelDataSchema:
    name: str
    required: dict                          # {col: dtype}, always present
    targets: dict = field(default_factory=dict)
    optional: dict = field(default_factory=dict)

    def columns(self, present, is_inference: bool) -> dict:
        present = set(present)
        cols = dict(self.required)
        if not is_inference:
            cols.update(self.targets)
        missing = sorted(set(cols) - present)
        if missing:
            mode = "inference" if is_inference else "training"
            raise ValueError(
                f"[{self.name}, {mode}] missing columns {missing}; present: {sorted(present)}"
            )
    
        if is_inference:
            cols.update({k: v for k, v in self.targets.items() if k in present})
        cols.update({k: v for k, v in self.optional.items() if k in present})
        return cols

    def check_required(self, data):
        missing = set(self.required) - data.keys()
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")


DHS = ModelDataSchema(
    name='DHS',
    required={
        "chrom": np.str_,
        "summit": np.int32,
        "sample_id": np.str_
    },
    targets={
        "read_depth": np.float32,
        "dhs_id": np.str_,
        "background": np.float32,
        "class": np.int8,
        "density": np.float32
    },
    optional={
        "indiv_id": np.str_,
        "shift": np.int32,
        "revcomp": np.bool_,
    }
)


VARIANT = ModelDataSchema(
    name='Variant',
    required={
        "chrom": np.str_,
        "pos": np.int32,
        "ref": np.str_,
        "alt": np.str_,
        "sample_id": np.str_
    },
    targets={
        "ref_counts": np.float32,
        "total_counts": np.float32,
        "BAD": np.float32,
        "logit_es": np.float32
    },
    optional={
        "indiv_id": np.str_,
        "shift": np.int32,
        "revcomp": np.bool_,
        "read_depth": np.float32,
        "background": np.float32,
        "density": np.float32
    }
)


def encode_inplace(
    data: dict,
    encodings: dict,
    key: str,
):
    """
    Encode categorical fields in `data` inplace.

    Parameters
    ----------
    data : dict
        Data dictionary. Values are arrays.
    encodings : dict
        Encoding dictionary to update inplace.
    key : str
        Key in `data` to encode.
    """
    arr = np.asarray(data[key], dtype=object)
    mask = pd.isna(arr) | np.isin(arr, ['None', 'nan'])
    arr = arr.copy()
    arr[mask] = ''
    enc, inv = np.unique(arr, return_inverse=True)
    data[key] = inv.astype(np.int32)
    encodings[key] = enc.astype(np.str_)


def sanitize_data(schema: ModelDataSchema, data: dict, encodings: dict = None, is_inference=False) -> tuple:
    """
        Ensure that all data arrays are contiguous and of the correct dtype. Columns not part of
        `schema` (required/targets/optional) are dropped from the result.
    """
    if encodings is None:
        encodings = {}
    keys = schema.columns(present=data, is_inference=is_inference)
    for key, dtype in keys.items():
        if dtype == np.str_:
            if key in encodings:
                data[key] = np.asarray(data[key], dtype=np.int32)
                encodings[key] = np.asarray(encodings[key], dtype=np.str_)
                mask = pd.isna(encodings[key]) | np.isin(encodings[key], ['None', 'nan'])
                encodings[key][mask] = ''
            else:
                encode_inplace(data, encodings, key)
        else:
            data[key] = np.asarray(data[key], dtype=dtype)
        if not data[key].flags["C_CONTIGUOUS"]:
            data[key] = np.ascontiguousarray(data[key])

    data = {key: data[key] for key in keys}

    if 'background' in data:
        data['background'] = np.nan_to_num(data['background'], copy=False)
    return data, encodings
