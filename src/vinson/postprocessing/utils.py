import anndata as ad
import pandas as pd
import numpy as np


def get_corrected_density(density, bg_density, min=0.01, max=20):
    return np.clip(
        density - np.clip(bg_density, min, None),
        min,
        max,
    )

def annotate_eval_dataset_with_layers(eval_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate the evaluation dataset with additional information from the AnnData object.

    Args:
        eval_dataset: The evaluation dataset to annotate. pd.DataFrame with columns:
            - 'pred_corrected_density'
            - 'y_hat'
            - 'density'
            - 'read_depth'
            - 'background'
            - 'sample_id'
    Returns:
        Modified eval_dataset with additional columns:
            - 'pred_counts'
            - 'pred_total_density'
            - 'target_counts'
            - 'bg_density'
            - 'corrected_density'
    """
    eval_dataset['bg_density'] = eval_dataset.eval('background * 1e6 / read_depth')
    eval_dataset['target_counts'] = eval_dataset.eval('density / 1e6 * read_depth')
    eval_dataset['corrected_density'] = get_corrected_density(
        eval_dataset['density'],
        eval_dataset['bg_density']
    )
    eval_dataset['pred_total_density'] = eval_dataset.eval('pred_corrected_density + bg_density')
    eval_dataset['pred_counts'] = eval_dataset.eval('pred_total_density / 1e6 * read_depth')
    return eval_dataset


def annotate_eval_dataset_with_obs_columns(eval_dataset: pd.DataFrame, adata: ad.AnnData, cols):
    for col in cols:
        eval_dataset[col] = eval_dataset['sample_id'].map(adata.obs[col])
    return eval_dataset
