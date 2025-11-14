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

def get_agg_by_annotation(df, column, by='extended_annotation'):
    gb = df.groupby(by).agg(
        median=(column, 'median'),
        q1=(column, lambda x: np.percentile(x, 25)),
        q3=(column, lambda x: np.percentile(x, 75)),
    )

    return gb

def calculate_per_dhs_fold_changes(
    eval_dataset: pd.DataFrame,
    cols: list,
    **kwargs
) -> pd.DataFrame:
    """
    """
    for col in cols:
        eval_dataset['_tmp_log_data'] = np.log2(eval_dataset[col].clip(kwargs.get('min', 0.005), kwargs.get('max', 20)))
        log_data = eval_dataset.pivot(index='dhs_id', columns='sample_id', values='_tmp_log_data')
        
        sample_by_annotation = pd.get_dummies(
            eval_dataset.drop_duplicates('sample_id').set_index('sample_id')['extended_annotation'], prefix='', prefix_sep=''
        ).astype(int)
        average_log_data = log_data @ sample_by_annotation / sample_by_annotation.sum(axis=0)
        
        eval_dataset[col + '_log2_fc'] = eval_dataset['_tmp_log_data'] - eval_dataset['dhs_id'].map(average_log_data.mean(axis=1))
    
        del eval_dataset['_tmp_log_data']

    return eval_dataset

def annotate_eval_dataset_with_obs_columns(eval_dataset: pd.DataFrame, adata: ad.AnnData, cols):
    for col in cols:
        eval_dataset[col] = eval_dataset['sample_id'].map(adata.obs[col])
    return eval_dataset

def get_samples_used_in_training_for_dhs(train_adata, dhs_ids):
    if len(dhs_ids) == 0:
        return np.array([])
    in_training = np.zeros(train_adata.shape[0], dtype=bool)
    for epoch_name in train_adata.uns['epoch_names']:
        example_class = train_adata[:, dhs_ids].layers[f'class.{epoch_name}']
        in_training |= example_class.toarray()[:, 0] != 0
    return train_adata.obs_names[in_training]

