import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def group_plot(g, column, type='box', order=None, ax=None, **kwargs):
    if order is not None:
        kwargs = {**kwargs, 'order': order, 'hue_order': order}

    n_total = g['extended_annotation'].nunique()
    n = n_total if order is None else len(order)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12 / n_total * n / 2.54, 2 / 2.54))
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
