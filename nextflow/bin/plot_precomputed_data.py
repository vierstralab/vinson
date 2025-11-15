import anndata as ad
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse
import scipy

from genome_tools.data.anndata import read_zarr_backed

from vinson.utils.data_formatting import extract_data_from_h5
from vinson.utils.helpers import read_configs
from vinson.postprocessing.plotting.per_dhs import obs_pred_barplot_by_ann, scatter_by_ann_train_val, aggregate_eval_dataset_by_sample
from vinson.postprocessing.utils import annotate_eval_dataset_with_layers, annotate_eval_dataset_with_obs_columns, calculate_per_dhs_fold_changes, get_samples_used_in_training_for_dhs
from vinson.postprocessing.metrics import calc_metrics

def cm2inch(value):
    """Function to convert cm to inches"""
    return value/2.54

def array2inch(*args):
    return tuple(cm2inch(x) for x in args)

def plot_density_correlation(
    eval_dataset: pd.DataFrame,
    x_col: str = "bg_corrected_density",
    y_col: str = "pred_total_density",
    max_points: int = 20_000,
    xlim: tuple = (0, 5),
    ylim: tuple = (0, 5),
    ax=None
):
    if ax is None:
        ax = plt.gca()
    """Plot hexbin correlation between predicted and observed densities."""
    df = eval_dataset.dropna(subset=[x_col, y_col]).copy()
    df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])]

    # Subsample for performance
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    x = df[x_col]
    y = df[y_col]

    pearson = scipy.stats.pearsonr(x, y)

    hb = ax.hexbin(x, y, bins="log", cmap="Blues", extent=(*xlim, *ylim))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axline((0, 0), slope=1, color="r", ls="--", lw=0.8)

    # Styling
    ax.set_xlabel("Observed corrected density")
    ax.set_ylabel("Predicted total density")
    ax.text(
        0.05, 0.9,
        f"R = {pearson.statistic:.2f}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize="small",
    )
    plt.colorbar(hb, ax=ax, label="log10(N)")
    return ax


def main(adata, eval_dataset: pd.DataFrame, output_prefix, annotation_data: pd.DataFrame, train_adata):
    eval_dataset = annotate_eval_dataset_with_layers(eval_dataset)
    eval_dataset = annotate_eval_dataset_with_obs_columns(
        eval_dataset,
        adata,
        ['SPOT3_score', 'alignment_quality', 'extended_annotation']
    )
    eval_dataset['extended_annotation'] = eval_dataset['extended_annotation'].astype(str)
    print('...')
    eval_dataset = calculate_per_dhs_fold_changes(eval_dataset, cols=['pred_corrected_density', 'corrected_density'])

    # barplots
    axes = obs_pred_barplot_by_ann(eval_dataset, 'corrected_density', 'pred_corrected_density', annotation_data.reset_index(),
                            error_kw=dict(linewidth=0, capthick=0, capsize=0))
    axes[0, 0].set_ylabel('Density\n(bg. corrected)')
    plt.gcf().suptitle(output_prefix)
    plt.savefig(f'{output_prefix}_corrected_density.pdf', transparent=True, bbox_inches='tight')
    plt.close(plt.gcf())
    
    axes = obs_pred_barplot_by_ann(eval_dataset, 'corrected_density_log2_fc', 'pred_corrected_density_log2_fc', annotation_data.reset_index(),)
                                # error_kw=dict(linewidth=0.25))
    axes[0, 0].set_ylabel('Density\n(log2 FC to DHS avg.)')
    plt.gcf().suptitle(output_prefix)
    plt.savefig(f'{output_prefix}_lfc.pdf', transparent=True, bbox_inches='tight')
    plt.close(plt.gcf())


    # per sample scatterplots
    val_ids = train_adata.var_names[train_adata.varm['split_data'] == 'val']
    val_eval_dataset = eval_dataset.query('dhs_id in @val_ids')

    fig, axes = plt.subplots(
        1, 2, figsize=array2inch(10, 5),
        sharex=True, sharey=True
    )

    tax = axes[0]
    metrics = []
    for dset, ax in zip([eval_dataset, val_eval_dataset], axes):
        gb = aggregate_eval_dataset_by_sample(dset, annotation_data.reset_index())
        metrics.append(calc_metrics(gb))
        in_training = get_samples_used_in_training_for_dhs(train_adata, dset['dhs_id'].unique())
        
        if len(gb) > 0:
            scatter_by_ann_train_val(gb, in_training, ax=ax)
            tax = ax
        else:
            ax.axis('off')
 
    t = tax.text(
        1,
        0.95,
        f"AUPRC = {metrics[0]['ap']:0.3f} ({metrics[1]['ap']:0.3f})\nAUROC = {metrics[0]['auroc']:0.3f} ({metrics[1]['auroc']:0.3f})\nPearson corr. (pos.) = {metrics[0]['pearson_pos']:0.3f} ({metrics[1]['pearson_pos']:0.3f})\nRank corr. (pos.) = {metrics[0]['spearman_pos']:0.3f} ({metrics[1]['spearman_pos']:0.3f})",
        fontdict=dict(fontsize="large"),
        transform=tax.transAxes,
        ha='left',
        va='top',
    )
    axes[0].set_title('All data')
    axes[1].set_title('Validation chromosomes')
    axes[0].set_ylabel("Prredicted density\n(bg. corrected)")
    axes[1].set_xlabel("Observed density\n(bg. corrected)")
    axes[1].set_xlabel("Observed density\n(bg. corrected)")
    plt.suptitle(output_prefix)
    plt.savefig(f"{output_prefix}_obs_pred_scatter.pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)


    # density correlation plots
    fig, ax = plt.subplots(figsize=array2inch(5, 5))
    plot_density_correlation(
        eval_dataset,
        x_col='corrected_density',
        y_col='pred_corrected_density',
        xlim=(0, 2),
        ylim=(0, 2),
        ax=ax
    )
    plt.savefig(f"{output_prefix}_corrected_density_correlation.pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)


if __name__ == '__main__':
    print('Visualizing prediction results')
    parser = argparse.ArgumentParser(description="Plot predictions")
    print('Adding options to parser')
    parser.add_argument("--prefix", type=str, help='Unique prefix for the model')
    parser.add_argument('--h5_data', help='h5 dataset prediction input')
    parser.add_argument('--npy_prediction', help='Path to model predictions (.npy file)')
    parser.add_argument('--annotation_data', help='Path to annotation data file (color and order for annotations)', default=None)
    parser.add_argument('--adata', help='Path to AnnData file with sample annotations', required=True)
    parser.add_argument('--train_adata', help='Path to AnnData file used in training', required=True)
    parser.add_argument('--output', help='Path to save visualizations', default='./')
    parser.add_argument('--model_config', help='Path to model config file')
    args = parser.parse_args()

    adata = read_zarr_backed(args.adata)
    train_adata = ad.read_h5ad(args.train_adata)

    output = f'{args.output}/{args.prefix}'
    if args.annotation_data is not None:
        annotation_plot_data = pd.read_table(args.annotation_data)
    else:
        raise ValueError('Annotation data file must be provided for plotting')

    eval_dataset, embeds = extract_data_from_h5(args.h5_data, adata) # Maybe embeds are not needed here
    log_output = read_configs(args.model_config)['model_kwargs'].get('log_output', False)

    if not log_output:
        eval_dataset['pred_corrected_density'] = np.load(args.npy_prediction)
        eval_dataset['pred_log_density'] = np.log(np.clip(eval_dataset['pred_corrected_density'], a_min=0.005, a_max=None))
    else:
        eval_dataset['pred_log_density'] = np.load(args.npy_prediction)
        eval_dataset['pred_corrected_density'] = np.exp(eval_dataset['pred_log_density'])

    eval_dataset = pd.DataFrame(eval_dataset)
    
    main(
        adata=adata,
        eval_dataset=eval_dataset,
        output_prefix=output,
        annotation_data=annotation_plot_data,
        train_adata=train_adata,
    )
