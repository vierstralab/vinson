import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from vinson.postprocessing.utils import get_agg_by_annotation


def get_agg_by_annotation_obs_pred(df, x_col, y_col, annotation_data, by=['dhs_id', 'extended_annotation']):
    x_gb = get_agg_by_annotation(df, x_col, by=by)
    y_gb = get_agg_by_annotation(df, y_col, by=by)
    gb = x_gb.merge(
        y_gb[['median', 'q1', 'q3']],
        left_index=True,
        right_index=True,
        suffixes=('_x', '_y')
    ).reset_index()

    gb['color'] = gb['extended_annotation'].map(
        annotation_data.set_index('extended_annotation')['color']
    ).fillna('#D0D0D0')
    return gb

def aggregate_eval_dataset_by_sample(eval_dataset, annotation_data):
    gb = get_agg_by_annotation_obs_pred(
        eval_dataset,
        x_col='corrected_density',
        y_col='pred_corrected_density',
        annotation_data=annotation_data,
        by=['sample_id', 'extended_annotation'],
    )
    return gb

def scatter_by_ann(
    gb,
    ax=None,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5 / 2.54, 5 / 2.54))

    kw = dict(
        s=1,
        alpha=1,
    )

    kw = {**kw, **kwargs}

    ax.scatter(
        gb['median_x'],
        gb['median_y'],
        c=kw.pop('color', gb['color']),
        edgecolor=kw.pop('edgecolor', gb['color']),
        **kw
    )

    ax.axline((0, 0), slope=1, color='k', lw=0.5, ls='--')

    return ax


def scatter_by_ann_train_val(gb, in_training, pos_tr=0.05, ax=None):
    if ax is None:
        ax = plt.gca()
    ax = scatter_by_ann(
        gb.query('sample_id not in @in_training'),
        s=3,
        lw=0.5,
        color='none',
        ax=ax,
    )
    ax = scatter_by_ann(
        gb.query('sample_id in @in_training'),
        s=5,
        marker='X',
        edgecolor='k',
        lw=0,
        ax=ax,
    )
    ax.axvline(pos_tr, color='k', lw=0.5, ls='--')
    return ax


def barplot_by_ann_with_offset(df, column, annotation_data, w=0.35, offset=0, ax=None, color='annotation', edgecolor='none', label=None, **kwargs):
    n = len(annotation_data)
    if ax is None:
        ax = plt.gca()

    gb = get_agg_by_annotation(df, column, by='extended_annotation').loc[annotation_data['extended_annotation']]
    gb['color'] = annotation_data['color'].fillna('#D0D0D0').values

    kw = dict(
        error_kw=dict(
            linewidth=0.5,
            capsize=1,
            capthick=0.5,
        ),
        linewidth=0,
    )
    kw['error_kw'] = {**kw['error_kw'], **kwargs.pop('error_kw', {})}
    kw = {**kw, **kwargs}

    for i, (_, gb_row) in enumerate(gb.iterrows()):
        ax.bar(
            i + offset * w,
            gb_row['median'],
            w, 
            yerr=[[gb_row['median'] - gb_row['q1']], [gb_row['q3'] - gb_row['median']]],
            color=gb_row['color'] if color == 'annotation' else color,
            edgecolor=gb_row['color'] if edgecolor == 'annotation' else edgecolor,
            label=label if i == 0 else None,
            **kw,
        )
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(annotation_data['name'], rotation=90, fontsize='small')
    margin = max(w * (abs(offset) + 0.5), 0.5)
    ax.set_xlim(-margin, n - 1 + margin)
    return ax


def obs_pred_barplot_by_ann(
        df, obs_col, pred_col, annotation_data,
        figsize_per_annotation=(12 / 33 / 2.54, 2 / 2.54),
        separate_axes=False,
        **kwargs
):
    n = len(annotation_data)
    figsize=(n * figsize_per_annotation[0], figsize_per_annotation[1] if not separate_axes else figsize_per_annotation[1] * 2)
    fig, axes = plt.subplots(1 if not separate_axes else 2, 1, figsize=figsize, squeeze=False)

    if not separate_axes:
        ax1 = axes[0, 0]
        ax2 = ax1
        offset = 0.5
    else:
        ax1 = axes[0, 0]
        ax2 = axes[1, 0]
        offset = 0

    barplot_by_ann_with_offset(
        df,
        obs_col,
        annotation_data,
        offset=-offset,
        color='#E7E7E7',
        edgecolor='annotation',
        linewidth=0.5,
        label='Observed',
        ax=ax1,
        **kwargs,
    )
    barplot_by_ann_with_offset(
        df,
        pred_col,
        annotation_data,
        offset=offset,
        color='annotation',
        edgecolor='annotation',
        linewidth=0.5,
        label='Predicted',
        ax=ax2,
        **kwargs,
    )

    if separate_axes:
        ax1.legend(frameon=False, fontsize='small', loc='upper right')
        ax1.set_xticks([])
    ax2.legend(frameon=False, fontsize='small', loc='upper right')
    return axes
