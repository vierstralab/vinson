import anndata as ad
import pandas as pd
import numpy as np
from scipy.stats import gmean


def get_corrected_density(density, bg_density, min=0.005, max=20):
    return np.clip(
        density - bg_density,
        min,
        max,
    )

def clipped_gmean(series, min=0.005, max=20):
    clipped = np.clip(series, min, max)
    return gmean(clipped)

def annotate_eval_dataset_with_layers(eval_dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Annotate the evaluation dataset with additional information from the AnnData object.

    Args:
        eval_dataset: The evaluation dataset to annotate. pd.DataFrame with columns:
            - 'pred_corrected_density'
            - 'density'
            - 'read_depth'
            - 'background'
            - 'sample_id'
    Returns:
        Modified eval_dataset with additional columns:
            - 'pred_counts'
            - 'pred_total_density'
            - 'counts'
            - 'bg_density'
            - 'corrected_density'
    """
    eval_dataset['bg_density'] = eval_dataset.eval('background * 1e6 / read_depth')
    eval_dataset['counts'] = eval_dataset.eval('density / 1e6 * read_depth')
    eval_dataset['corrected_density'] = get_corrected_density(
        eval_dataset['density'],
        eval_dataset['bg_density'],
        **kwargs
    )
    eval_dataset['pred_total_density'] = eval_dataset.eval('pred_corrected_density + bg_density')
    eval_dataset['pred_counts'] = eval_dataset.eval('pred_total_density / 1e6 * read_depth')
    return eval_dataset


def calculate_per_dhs_fold_changes(
    eval_dataset: pd.DataFrame,
    col: str = 'corrected_density',
    **kwargs
) -> pd.Series:
    """
    """
    
    per_dhs_and_annotation_metrics = eval_dataset.groupby(
        ['dhs_id', 'extended_annotation']
    )[col].agg(clipped_gmean, **kwargs).reset_index()
    per_dhs_mean = per_dhs_and_annotation_metrics.groupby('dhs_id')[col].transform(clipped_gmean, **kwargs)

    return (eval_dataset.set_index('dhs_id')[col] / per_dhs_mean).set_index(eval_dataset.index) # <- Series


def annotate_eval_dataset_with_obs_columns(eval_dataset: pd.DataFrame, adata: ad.AnnData, cols):
    for col in cols:
        eval_dataset[col] = eval_dataset['sample_id'].map(adata.obs[col])
    return eval_dataset
