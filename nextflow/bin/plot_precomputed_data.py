import seaborn as sns
import scipy
from matplotlib import rcParams
import sys
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def cm2inch(value):
    """Function to convert cm to inches"""
    return value/2.54

def array2inch(*args):
    return tuple(cm2inch(x) for x in args)
import argparse
from genome_tools.data.anndata import read_zarr_backed
rcParams["font.sans-serif"] = ["IBM Plex Sans"]
import configparser


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

def df_from_h5(path):
    with h5py.File(path, 'r') as f:
        data = {}
        for key in f.keys():
            arr = f[key][:]
            # Decode byte strings if needed
            if arr.dtype.kind == 'S':
                arr = arr.astype(str)
            data[key] = arr
        return pd.DataFrame(data)

def main(prefix, dataset, predict_output, output_dir):
    #read in dataset and np array of predictions
    params = configparser.ConfigParser()
    params.read("/home/mbrannon/ENCODE4_DHS_index/common/release_paths.ini")

    zarr_human_path = params['index']['human_anndata']
    human_data = read_zarr_backed(zarr_human_path)
    vinson_training_data = human_data[
        human_data.obs.eval('pathological_state == "Normal" & SPOT3_score >= 0.1'),
        human_data.var.eval('autosomal_dhs'),]
    
    dnase_nmf_config = configparser.ConfigParser()
    dnase_nmf_config.read(params['embedding_nmf']['config_file'])
    component_data = pd.read_table(dnase_nmf_config['NMF']['component_data'])

    eval_dataset = df_from_h5(dataset)
    eval_dataset['y_hat'] = np.load(predict_output)
    
    #calcualte results
    eval_dataset['pred_counts'] = eval_dataset.eval('exp(y_hat) / 1e6 * read_depth + background')
    eval_dataset['target_counts'] = eval_dataset.eval('density / 1e6 * read_depth')
    eval_dataset['bg_density'] = eval_dataset.eval('background * 1e6 / read_depth')
    eval_dataset['bg_corrected_density'] = np.clip(eval_dataset.eval('density - bg_density'), 0, None)
    eval_dataset['pred_corrected_density'] = eval_dataset.eval('exp(y_hat)')
    eval_dataset['pred_total_density'] = eval_dataset.eval('exp(y_hat) + bg_density')
    
    #categorize
    eval_dataset['extended_annotation'] = eval_dataset['sample_id'].map(vinson_training_data.obs['extended_annotation'].to_dict())
    eval_dataset['core_annotation'] = eval_dataset['sample_id'].map(vinson_training_data.obs['core_annotation'].to_dict())
    eval_dataset['system'] = eval_dataset['sample_id'].map(vinson_training_data.obs['system'].to_dict())

    #group by category
    g = eval_dataset.groupby(['dhs_id', 'extended_annotation', 'core_annotation', 'system'])[[
        'bg_corrected_density', 'y_hat', 'pred_corrected_density', 'density', 'pred_total_density',
    ]].mean().reset_index()
    
    #calculate metrics by group
    g['log_dens'] = np.log(g['bg_corrected_density'] + 1e-1)
    g['log_dens_mean'] = g.groupby('dhs_id')['log_dens'].transform('mean')
    g['log_dens_lfc'] = g.eval('log_dens - log_dens_mean') # Just another way to plot it, instead of just density
    g['dens_delta'] = g.eval('exp(log_dens_lfc)')

    g['y_hat_mean'] = g.groupby('dhs_id')['y_hat'].transform('mean')
    g['y_hat_lfc'] = g.eval('y_hat - y_hat_mean')
    g['y_hat_delta'] = g.eval('exp(y_hat_lfc)')

    gb = g.groupby("extended_annotation").agg(
        log_dens_lfc=('log_dens_lfc', 'median'),
        core_annotation=('core_annotation', 'first'),
        system=('system', 'first'),
    ).sort_values(
        ['system', 'core_annotation', 'log_dens_lfc']
    )
    ## change later
    gb['short_name'] = gb.index

    gb['order'] = gb['short_name'].map(
    component_data.reset_index(names=['order']).set_index('short_name')['order'].to_dict()
    )
    gb['color'] = gb['short_name'].map(
        component_data.set_index('short_name')['color'].to_dict()
    )
    gb = gb.sort_values('order').dropna()
    order = gb.index
    palette = gb['color'].to_dict()

    n_total = g['extended_annotation'].nunique()
    n = n_total if order is None else len(order)

    fig, axes = plt.subplots(2, 1, figsize=array2inch(12 / n_total * n, 4))
    for i, col in enumerate(['y_hat_lfc', 'log_dens_lfc']):
        ax = axes[i]
        ax = group_plot(g, col, type='bar', order=order, palette=palette, ax=ax)
        ax.set_ylabel(col)
        ax.axhline(0, color='grey', zorder=1e6, ls='--')
        ax.set_ylim(-2, 2)
        if i == 0:
            ax.set_xticks([])
            ax.set_xlabel('')
    plt.savefig(f'{output_dir}/{prefix}_lfc.pdf', transparent=True, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=array2inch(12 / n_total * n, 4))
    for i, col in enumerate(['pred_corrected_density', 'bg_corrected_density']):
        ax = axes[i]
        ax = group_plot(g, col, type='bar', order=order, palette=palette, ax=ax)
        ax.set_ylabel(col)
        ax.axhline(0, color='grey', zorder=1e6, ls='--')
        ax.set_ylim(0, 2)
        if i == 0:
            ax.set_xticks([])
            ax.set_xlabel('')
    plt.savefig(f'{output_dir}/{prefix}_corrected_density.pdf', transparent=True, bbox_inches='tight')

if __name__ == '__main__':
    print('Visualizing prediction results')
    parser = argparse.ArgumentParser(description="Plot predictions")
    print('Adding options to parser')
    parser.add_argument("--prefix", type=str, help='sample indicator name')
    parser.add_argument('--dataset', help='h5 dataset prediction input')
    parser.add_argument('--predict-output', help='predictions output from predict process, numpy')
    parser.add_argument('--output-dir', help='Path to save visualizations', default='./')
    args = parser.parse_args()
    main(args.prefix, args.dataset, args.predict_output, args.output_dir)
