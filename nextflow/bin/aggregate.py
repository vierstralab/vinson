import sys
import h5py
import numpy as np
import pandas as pd


h5_file = '/net/seq/data2/projects/mbrannon/work/60/2555c0e24ac672256e4508e7ba2e7f/Lymphoid_variants.h5'
pred_file = '/net/seq/data2/projects/mbrannon/work/60/2555c0e24ac672256e4508e7ba2e7f/Lymphoid_variants.npy'
group_name = 'Lymphoid'
outfile = '/net/seq/data2/projects/mbrannon/work/60/2555c0e24ac672256e4508e7ba2e7f/lymphoid_filtered_var.parquet'


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


def aggregate_effect_size(es, weights):
    return np.average(es, weights=weights)
agg_df = 

agg_df = (
    df
    .groupby(
        ["chrom", "pos", "ref", "alt", "group_id"],
        as_index=False
    )
    .apply(
        lambda x: pd.Series({
            "agg_group_es":
                aggregate_effect_size(
                    x["es"],
                    x["inverse_mse"]
                ),
            "agg_pred_logit_es":
                aggregate_effect_size(
                    x["pred"],
                    x["inverse_mse"]
                ),
            "agg_logit_group_es":
                aggregate_effect_size(
                    x["logit_es"],
                    x["inverse_mse"]
                ),
            "min_FDR_sample":
                x["FDR_sample"].min(),
            "median_FDR_sample":
                x["FDR_sample"].median(),
            "n_samples":
                x["sample_id"].nunique(),
            "sum_weights":
                x["inverse_mse"].sum(),
        })
    )
    .reset_index(drop=True)
)

agg_df.to_parquet(outfile, index=False)