import anndata as ad
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse

from genome_tools.data.anndata import read_zarr_backed

from vinson.utils.data_formatting import extract_data_from_h5


def get_palette_dict(categories):
    pass

def cm2inch(value):
    """Function to convert cm to inches"""
    return value/2.54

def array2inch(*args):
    return tuple(cm2inch(x) for x in args)


def group_plot(g, column, type='box', order=None, ax=None, **kwargs):
    if order is not None:
        kwargs = {**kwargs, 'order': order, 'hue_order': order}

    n_total = g['extended_annotation'].nunique()
    n = n_total if order is None else len(order)
    if ax is None:
        fig, ax = plt.subplots(figsize=array2inch(12 / n_total * n, 2))
    if type == 'box':
        sns.stripplot(data=g, x='extended_annotation', y=column, s=1, ax=ax, hue='extended_annotation', alpha=0.5, linewidth=0, **kwargs)
        sns.boxplot(data=g, x='extended_annotation', y=column, ax=ax, hue='extended_annotation', fill=False, color='k', linewidth=0.5, showfliers=False, zorder=1e6, **kwargs)
    elif type == 'bar':
        gb = g.groupby('extended_annotation').agg(
            median=(column, 'median'),
            q1=(column, lambda x: np.percentile(x, 25)),
            q3=(column, lambda x: np.percentile(x, 75)),
        )
        sns.barplot(
            data=gb,
            x='extended_annotation',
            y='median',
            ax=ax,
            hue='extended_annotation',
            edgecolor='k',
            linewidth=0.5,
            errorbar=None,
            **kwargs
        )
        if order is not None:
            gb = gb.reindex(order)
        ax.errorbar(
            x=np.arange(gb.shape[0]),
            y=gb['median'],
            yerr=[gb['median'] - gb['q1'], gb['q3'] - gb['median']],
            fmt='none',
            ecolor='k',
            elinewidth=0.5,
            capsize=1,
            capthick=0.5,
            zorder=1e6
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize='small')
    ax.set_xlim(-0.5, n - 0.5)
    return ax

def plot_per_annotation_comparison(df: pd.DataFrame, annotation_data: pd.DataFrame, pred_col, target_col):
    per_annotation_metrics = df.groupby(["extended_annotation"]).agg(
        log_dens_lfc=('log_dens_lfc', 'median'),
        core_annotation=('core_annotation', 'first'),
        system=('system', 'first'),
    ).sort_values(
        ['system', 'log_dens_lfc']
    )
    fig, axes = plt.subplots(2, 1, figsize=array2inch(12 / n_total * n, 4))
    for i, column in enumerate([pred_col, target_col]):
        ax = axes[i]
        ax = group_plot(df, column, type='bar', order=annotation_data['index'], palette=annotation_data['color'], ax=ax)
        ax.set_ylabel(column)
        ax.axhline(0, color='grey', zorder=1e6, ls='--')
        ax.set_ylim(-2, 2)
        if i == 0:
            ax.set_xticks([])
            ax.set_xlabel('')
    return axes


def get_mock_annotation_data(anndata, annotation_column='extended_annotation'):
    annotation_data = pd.DataFrame({
        'name': anndata.obs[annotation_column].unique(),
    })
    annotation_data['short_name'] = annotation_data['name']
    annotation_data['index'] = np.arange(len(annotation_data))
    annotation_data['color'] = annotation_data['name'].map(get_palette_dict(annotation_data['name']))
    return annotation_data


def annotate_eval_dataset(eval_dataset: pd.DataFrame, adata: ad.AnnData) -> pd.DataFrame:
    eval_dataset['pred_counts'] = eval_dataset.eval('exp(y_hat) / 1e6 * read_depth + background')
    eval_dataset['target_counts'] = eval_dataset.eval('density / 1e6 * read_depth')
    eval_dataset['bg_density'] = eval_dataset.eval('background * 1e6 / read_depth')
    eval_dataset['bg_corrected_density'] = np.clip(eval_dataset.eval('density - bg_density'), 0, None)
    eval_dataset['pred_corrected_density'] = eval_dataset.eval('exp(y_hat)')
    eval_dataset['pred_total_density'] = eval_dataset.eval('exp(y_hat) + bg_density')
    eval_dataset['extended_annotation'] = eval_dataset['sample_id'].map(adata.obs['extended_annotation'].to_dict())
    eval_dataset['core_annotation'] = eval_dataset['sample_id'].map(adata.obs['core_annotation'].to_dict())
    eval_dataset['system'] = eval_dataset['sample_id'].map(adata.obs['system'].to_dict())
    return eval_dataset

def main(adata, eval_dataset: pd.DataFrame, output_prefix, annotation_data: pd.DataFrame):
    eval_dataset = annotate_eval_dataset(eval_dataset, adata)

    # group by annotation
    per_dhs_and_annotation_metrics = eval_dataset.groupby(
        ['dhs_id', 'extended_annotation', 'core_annotation', 'system']
    )[[
        'bg_corrected_density', 'y_hat', 'pred_corrected_density', 
        'density','pred_total_density',
    ]].mean().reset_index()

    pseudocount = 1e-2

    per_dhs_and_annotation_metrics['y_hat_mean'] = per_dhs_and_annotation_metrics.groupby('dhs_id')['y_hat'].transform('mean')
    per_dhs_and_annotation_metrics['y_hat_lfc'] = per_dhs_and_annotation_metrics.eval('y_hat - y_hat_mean')
    per_dhs_and_annotation_metrics['y_hat_delta'] = per_dhs_and_annotation_metrics.eval('exp(y_hat_lfc)')
    # per_dhs_and_annotation_metrics['log_pred_corrected_density'] = np.log(per_dhs_and_annotation_metrics['pred_corrected_density'] + pseudocount)
    # per_dhs_and_annotation_metrics['log_density'] = np.log(per_dhs_and_annotation_metrics['density'] + pseudocount)
    # per_dhs_and_annotation_metrics['log_pred_total_density'] = np.log(per_dhs_and_annotation_metrics['pred_total_density'] + pseudocount)
    # per_dhs_and_annotation_metrics['log_bg_corrected_density'] = np.log(per_dhs_and_annotation_metrics['bg_corrected_density'] + pseudocount)
    # per_dhs_and_annotation_metrics['log_y_hat'] = np.log(per_dhs_and_annotation_metrics['y_hat'] + pseudocount)
    per_dhs_and_annotation_metrics['median_log_dens_lfc'] = per_dhs_and_annotation_metrics.groupby('extended_annotation')['y_hat_lfc'].transform('median')

    #calculate metrics by group
    # g['log_dens'] = np.log(g['bg_corrected_density'] + 1e-1)
    # g['log_dens_mean'] = g.groupby('dhs_id')['log_dens'].transform('mean')
    # g['log_dens_lfc'] = g.eval('log_dens - log_dens_mean') # Just another way to plot it, instead of just density
    # g['dens_delta'] = g.eval('exp(log_dens_lfc)')

    # g['y_hat_mean'] = g.groupby('dhs_id')['y_hat'].transform('mean')
    # g['y_hat_lfc'] = g.eval('y_hat - y_hat_mean')
    # g['y_hat_delta'] = g.eval('exp(y_hat_lfc)')

    per_annotation_metrics = per_dhs_and_annotation_metrics.groupby(["core_annotation", "extended_annotation"]).agg(
        log_dens_lfc=('log_dens_lfc', 'median'),
        core_annotation=('core_annotation', 'first'),
        system=('system', 'first'),
    ).sort_values(
        ['system', 'log_dens_lfc']
    )
    per_annotation_metrics = annotation_data.set_index('name').join(
        per_annotation_metrics
    ).sort_values('index')

    n_total = g['extended_annotation'].nunique()

    axes = plot_per_annotation_comparison(
        per_dhs_and_annotation_metrics,
        pred_col='y_hat_lfc',
        target_col='log_dens_lfc',
        palette=palette,
        order=order
    )

    plt.savefig(f'{output_prefix}_lfc.pdf', transparent=True, bbox_inches='tight')
    plt.show()

    #fig, axes = plt.subplots(2, 1, figsize=array2inch(12 / n_total * n, 4))
    axes = plot_per_annotation_comparison(
        g,
        pred_col='pred_corrected_density',
        target_col='bg_corrected_density',
        palette=palette,
        order=order
    )
    plt.savefig(f'{output_prefix}_corrected_density.pdf', transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    print('Visualizing prediction results')
    parser = argparse.ArgumentParser(description="Plot predictions")
    print('Adding options to parser')
    parser.add_argument("prefix", type=str, help='Unique prefix for the model')
    parser.add_argument('h5_data', help='h5 dataset prediction input')
    parser.add_argument('npy_prediction', help='Path to model predictions (.npy file)')
    parser.add_argument('--annotation_data', help='Path to annotation data file (color and order for annotations)', default=None)
    parser.add_argument('--adata', help='Path to AnnData file with sample annotations', required=True)
    parser.add_argument('--output', help='Path to save visualizations', default='./')
    args = parser.parse_args()

    adata = read_zarr_backed(args.adata)

    output = f'{args.output}/{args.prefix}'
    if args.annotation_data is not None:
        annotation_plot_data = pd.read_table(args.annotation_data)
    else:
        annotation_plot_data = get_mock_annotation_data(adata)

    eval_dataset = extract_data_from_h5(args.h5_data)
    eval_dataset['y_hat'] = np.load(args.npy_prediction)
    eval_dataset = pd.DataFrame(eval_dataset)
    
    main(
        adata=adata,
        eval_dataset=eval_dataset,
        output_prefix=output,
        annotation_data=annotation_plot_data
    )
