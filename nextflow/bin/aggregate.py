import sys
import h5py
import numpy as np
import pandas as pd


h5_file = sys.argv[1]
pred_file = sys.argv[2]
group_name = sys.argv[3]
outfile = sys.argv[4]
aggregation_type = sys.argv[5]
annotation_file = sys.argv[6]


def load_h5_as_df(h5_file):
    with h5py.File(h5_file, "r") as f:
        data = {}

        for key in f.keys():
            arr = f[key][()]

            if hasattr(arr, "dtype") and arr.dtype.kind == "S":
                arr = arr.astype(str)

            data[key] = arr

    return pd.DataFrame(data)


def aggregate_effect_size(x, column):

    values = x[column].to_numpy()

    if aggregation_type == "weighted":

        weights = x["inverse_mse"].to_numpy()

        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0)
        )

        values = values[valid]
        weights = weights[valid]

        if len(values) == 0:
            return np.nan

        return np.average(values, weights=weights)

    else:
        return np.mean(values)


# ---------------------------------------------------------
# Load predictions
# ---------------------------------------------------------

df = load_h5_as_df(h5_file)

preds = np.load(pred_file).reshape(-1)

if len(df) != len(preds):
    raise ValueError(
        f"{group_name}: "
        f"{len(df)} rows but {len(preds)} predictions"
    )

df["pred"] = preds
df["group_id"] = group_name


# ---------------------------------------------------------
# Determine tested samples
#
# For "all", we do not need tested information at all.
#
# For "tested"/"weighted":
#   1. Use existing tested column if available
#   2. Otherwise use existing hotspots column
#   3. Otherwise load hotspots from annotation file
# ---------------------------------------------------------

if aggregation_type in ["tested", "weighted"]:

    if "tested" in df.columns:

        df["tested"] = df["tested"].astype(bool)

    elif "hotspots" in df.columns:

        df["tested"] = (
            df["hotspots"]
            .astype(str)
            .isin(["1", "-"])
        )

    else:

        if not annotation_file:
            raise ValueError(
                f"{group_name}: aggregation type '{aggregation_type}' "
                "requires tested/hotspots information, but no "
                "annotation_file was provided."
            )

        annotation = pd.read_parquet(annotation_file)

        if "hotspots" not in annotation.columns:
            raise ValueError(
                f"{group_name}: annotation file does not contain "
                "'hotspots'."
            )

        annotation = annotation[
            [
                "group_id",
                "sample_id",
                "chrom",
                "pos",
                "ref",
                "alt",
                "hotspots",
            ]
        ].copy()

        df = df.merge(
            annotation,
            on=[
                "group_id",
                "sample_id",
                "chrom",
                "pos",
                "ref",
                "alt",
            ],
            how="left",
        )

        df["tested"] = (
            df["hotspots"]
            .astype(str)
            .isin(["1", "-"])
        )


# ---------------------------------------------------------
# Select aggregation subset
# ---------------------------------------------------------

if aggregation_type == "all":

    agg_input = df.copy()

elif aggregation_type == "tested":

    agg_input = df[df["tested"]].copy()

elif aggregation_type == "weighted":

    agg_input = df[df["tested"]].copy()

else:

    raise ValueError(
        f"Unknown aggregation type: {aggregation_type}. "
        "Expected one of: all, tested, weighted"
    )


# ---------------------------------------------------------
# Aggregate
# ---------------------------------------------------------

agg_df = (
    agg_input
    .groupby(
        ["chrom", "pos", "ref", "alt", "group_id"],
        as_index=False
    )
    .apply(
        lambda x: pd.Series({

            "agg_group_es":
                aggregate_effect_size(
                    x,
                    "es"
                ),

            "agg_pred_logit_es":
                aggregate_effect_size(
                    x,
                    "pred"
                ),

            "agg_logit_group_es":
                aggregate_effect_size(
                    x,
                    "logit_es"
                ),

            "min_FDR_sample":
                x["FDR_sample"].min(),

            "median_FDR_sample":
                x["FDR_sample"].median(),

            "n_samples":
                x["sample_id"].nunique(),

            "sum_weights":
                x["inverse_mse"].sum(),

            "n_tested":
                x["tested"].sum()
                if "tested" in x.columns
                else np.nan,

        })
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

agg_df.to_parquet(outfile, index=False)