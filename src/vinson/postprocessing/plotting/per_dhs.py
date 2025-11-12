import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def obs_pred_barplot_by_ann(
        df, obs_col, pred_col, annotation_data, ax=None, **kwargs
):
    n = len(annotation_data)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12 / 33 * n / 2.54, 2 / 2.54))

    barplot_by_ann_with_offset(
        df,
        obs_col,
        annotation_data,
        offset=-0.5,
        color='grey',
        label='Observed',
        ax=ax,
        **kwargs,
    )
    barplot_by_ann_with_offset(
        df,
        pred_col,
        annotation_data,
        offset=0.5,
        label='Predicted',
        ax=ax,
        **kwargs,
    )

    ax.legend(frameon=False, fontsize='small', loc='upper right')

    return ax


def barplot_by_ann_with_offset(df, column, annotation_data, w=0.35, offset=0, ax=None, color=None, label=None, **kwargs):
    n = len(annotation_data)
    if ax is None:
        ax = plt.gca()

    gb = df.groupby('extended_annotation').agg(
        median=(column, 'median'),
        q1=(column, lambda x: np.percentile(x, 25)),
        q3=(column, lambda x: np.percentile(x, 75)),
    ).loc[annotation_data['extended_annotation']]

    gb['color'] = annotation_data['color'].values

    kw = dict(
        error_kw=dict(
            linewidth=0.5,
            capsize=1,
            capthick=0.5,
            fmt='none',
        ),
        linewidth=0,
        edgecolor='k',
    )
    kw = {**kw, **kwargs}

    for i, (_, gb_row) in enumerate(gb.iterrows()):
        ax.bar(
            i + offset * w,
            gb_row['median'],
            w, 
            yerr=[[gb_row['median'] - gb_row['q1']], [gb_row['q3'] - gb_row['median']]],
            color=gb_row['color'] if color is None else color,
            label=label if i == 0 else None,
            **kw,
        )

    ax.set_xticklabels(annotation_data['name'], rotation=90, fontsize='small')
    margin = max(w * (abs(offset) + 0.5), 0.5)
    ax.set_xlim(-margin, n - 1 + margin)
    return ax
