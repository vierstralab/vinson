import numpy as np
import pandas as pd


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
    arr = data[key]
    enc, inv = np.unique(arr, return_inverse=True)
    data[key] = inv.astype(np.int32)
    encodings[key] = enc.astype(np.str_)


def sanitize_data(data: dict, encodings: dict = None, is_variant=False) -> tuple:
    """Ensure that all data arrays are contiguous and of the correct dtype."""
    if encodings is None:
        encodings = {}
    if is_variant:
        data_keys = {
            "chrom": np.str_,
            "pos": np.int32,
            "ref": np.str_,
            "alt": np.str_,
            "ref_counts": np.float32,
            "total_counts": np.float32,
            "BAD": np.float32,
            "sample_id": np.str_,
            "logit_es": np.float32,
        }
    else:
        data_keys = {
            "read_depth": np.float32,
            "sample_id": np.str_,
            "dhs_id": np.str_,
            "chrom": np.str_,
            "summit": np.int32,
            "background": np.float32,
            "class": np.int8,
            "density": np.float32,
        }
    optional_keys = {
        "indiv_id": np.str_,
        "dhs_weight": np.float32,
    }
    keys = {
        **data_keys,
        **{x: y for x, y in optional_keys.items() if x in data},
    }
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

    if 'background' in data:
        data['background'] = np.nan_to_num(data['background'], copy=False)
    return data, encodings
