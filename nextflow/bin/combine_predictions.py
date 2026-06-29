import sys
import h5py
import numpy as np
import pandas as pd

h5_file = sys.argv[1]
pred_file = sys.argv[2]
group_name = sys.argv[3]
outfile = sys.argv[4]


def load_h5_as_df(h5_file):
    with h5py.File(h5_file, "r") as f:
        data = {}

        for key in f.keys():
            arr = f[key][()]

            if hasattr(arr, "dtype") and arr.dtype.kind == "S":
                arr = arr.astype(str)

            data[key] = arr

    return pd.DataFrame(data)


df = load_h5_as_df(h5_file)

preds = np.load(pred_file).reshape(-1)

if len(df) != len(preds):
    raise ValueError(
        f"{group_name}: "
        f"{len(df)} rows but {len(preds)} predictions"
    )

df["pred"] = preds
df["group_id"] = group_name


df.to_parquet(outfile, index=False)