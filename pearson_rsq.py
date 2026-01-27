import numpy as np
import pandas as pd
import scipy.stats as stats
import anndata as ad

from genome_tools.data.anndata import read_zarr_backed
from vinson.utils.data_formatting import extract_data_from_h5
from vinson.utils.helpers import read_configs
from vinson.postprocessing.utils import (
    annotate_eval_dataset_with_layers,
    annotate_eval_dataset_with_obs_columns,
    calculate_per_dhs_fold_changes,
)


# --------------------------
# FUNCTIONS
# --------------------------

def compute_per_sample_metrics(eval_dataset: pd.DataFrame):
    """
    Compute Pearson correlation and R² per sample_id in eval_dataset.
    Returns a DataFrame with columns: sample_id, pearson, r2
    """

    df = eval_dataset.dropna(
        subset=["corrected_density", "pred_corrected_density", "sample_id"]
    )

    def _calc(group):
        x = group["corrected_density"].astype(float)
        y = group["pred_corrected_density"].astype(float)

        if len(x) < 3:
            return pd.Series({"pearson": np.nan, "r2": np.nan})

        r, _ = stats.pearsonr(x, y)
        return pd.Series({"pearson": r, "r2": r * r})

    return df.groupby("sample_id").apply(_calc).reset_index()



def build_eval_dataset_from_row(row):
    """
    Given a row from results_df, load everything needed 
    and return eval_dataset as a DataFrame.
    """

    # Load AnnData
    adata = read_zarr_backed(adata_f)

    # Load model config
    cfg = read_configs(row["model_config"])
    log_output = cfg["model_kwargs"].get("log_output", False)

    # Load predictions
    preds = np.load(row["result_np"])

    # Build eval_dataset from stored h5 data
    eval_dataset, _ = extract_data_from_h5(row["dhs_dataset"], adata)
    eval_dataset = pd.DataFrame(eval_dataset)

    # Add predictions
    if log_output:
        eval_dataset["pred_log_density"] = preds
        eval_dataset["pred_corrected_density"] = np.exp(preds)
    else:
        eval_dataset["pred_corrected_density"] = preds
        eval_dataset["pred_log_density"] = np.log(
            np.clip(preds, a_min=0.005, a_max=None)
        )

    # Annotate
    eval_dataset = annotate_eval_dataset_with_layers(eval_dataset)

    # These annotations require matching metadata in adata
    cols = ["SPOT3_score", "alignment_quality", "extended_annotation"]
    eval_dataset = annotate_eval_dataset_with_obs_columns(eval_dataset, adata, cols)
    eval_dataset["extended_annotation"] = eval_dataset["extended_annotation"].astype(str)

    # Fold changes
    eval_dataset = calculate_per_dhs_fold_changes(
        eval_dataset,
        cols=["pred_corrected_density", "corrected_density"]
    )

    return eval_dataset



# --------------------------
# MAIN LOOP
# --------------------------
import pandas as pd
results_df=pd.read_table("/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/vinson_model/2025_11_04_trusting_goldwasser/predictions/output/per_sample_predictions_meta.w_checkpoint.annotated_with_predictions.tsv")
annotation_data = "/net/seq/data2/projects/ENCODE4Plus/REGULOME/sequence_to_accessibility_model/cell_selective/annotation_data.tsv"
adata_f = "/net/seq/data2/projects/ENCODE4Plus/REGULOME/one_big_beautiful_index/latest.reference_anndata.zarr"

pearsons = []
r2s = []

for idx, row in results_df.iterrows():
    print(f"Processing sample: {row['sample_id']} ({idx})")

    eval_dataset = build_eval_dataset_from_row(row)

    # Compute metrics
    per_sample_df = compute_per_sample_metrics(eval_dataset)

    # Expect one row since sample_id should be unique
    if len(per_sample_df) > 0:
        pearson = per_sample_df["pearson"].iloc[0]
        r2 = per_sample_df["r2"].iloc[0]
    else:
        pearson = np.nan
        r2 = np.nan

    pearsons.append(pearson)
    r2s.append(r2)

# Add to results_df
results_df["pearson"] = pearsons
results_df["r2"] = r2s


results_df.to_csv('/home/mbrannon/tmp/new_vinson_results.csv', index=False)