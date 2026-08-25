import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec, transforms
from matplotlib import colors as mcolors
from genome_tools.plotting import signal_plot
import genome_tools.plotting.colors.cm as cm
from .plotting import PlotComponent

from genome_tools.plotting.sequence import seq_plot


class ImbalancePlotComponent(PlotComponent):
    def __init__(self, use_stripplot=False, **kwargs):
        super().__init__(**kwargs)
        self.use_stripplot = use_stripplot  # optional toggle

    def plot(self, data, gs, fig):
        tested_df, anova_df = data

        ref_allele = anova_df['ref'].values[0]
        alt_allele = anova_df['alt'].values[0]
        rsid = anova_df['variant_id'].values[0]

        ax = fig.add_subplot(gs)

        # ---------------------------------------
        # OPTIONAL: raw points (actual data)
        # ---------------------------------------
        if self.use_stripplot:
            sns.stripplot(
                data=tested_df,
                y='group_name',
                x='logit_es',
                order=anova_df['group_name'],
                ax=ax,
                s=5,
                zorder=-1,
                orient='h',
                palette=self.names_to_colors,
                alpha=0.3
            )

        # ---------------------------------------
        # Predicted values
        # ---------------------------------------
        # Observed group effect from individual samples
        # x = anova_df['logit_es']
        # xerr = anova_df.get('logit_es_std', None)
        x = anova_df["logit_group_es"]
        xerr = anova_df["logit_es_std"]
        
        # Error bars
        if xerr is not None:
            ax.errorbar(
                x=x,
                y=anova_df['group_name'],
                xerr=xerr,
                fmt='.',
                capsize=4,
                capthick=1.0,
                c='k',
                lw=1.0
            )

        # Points
        ax.scatter(
            y=anova_df['group_name'],
            x=x,
            lw=3,
            ec=anova_df['group_name'].map(self.names_to_colors).values,
            color='k',
            s=12
        )

        # ---------------------------------------
        # Formatting
        # ---------------------------------------
        ax.set_yticklabels([
            x.get_text().replace(' (', '\n(').replace('/', '/\n')
            for x in ax.get_yticklabels()
        ])

        trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)

        ax.annotate(
            alt_allele,
            xy=(0.1, 1.05),
            xycoords=trans,
            fontname="IBM Plex Mono",
            fontweight="bold",
            fontsize=7,
            bbox=dict(boxstyle="Circle", ec='none',
                      fc=cm.map_vocab_color(alt_allele, 'dna')[0])
        )

        ax.annotate(
            ref_allele,
            xy=(0.9, 1.05),
            xycoords=trans,
            fontname="IBM Plex Mono",
            fontweight="bold",
            fontsize=7,
            bbox=dict(boxstyle="Circle", ec='none',
                      fc=cm.map_vocab_color(ref_allele, 'dna')[0])
        )

        # Horizontal separators
        for i in range(len(anova_df['group_name']) - 1):
            ax.axhline(i + 0.5, ls='--', color='k', lw=0.5)

        # ---------------------------------------
        # Axis limits (robust to missing std)
        # ---------------------------------------
        x1 = anova_df["logit_group_es"]
        e1 = anova_df["logit_es_std"].fillna(0)
        
        x2 = anova_df["pred_logit_es"]
        e2 = anova_df["pred_logit_es_std"].fillna(0)
        
        xmin = min(
            (x1-e1).min(),
            (x2-e2).min()
        )
        
        xmax = max(
            (x1+e1).max(),
            (x2+e2).max()
        )
        
        xlim = (-max(abs(xmin), abs(xmax))*1.3,
                max(abs(xmin), abs(xmax))*1.3)
        if xerr is not None:
            xmin = (x - xerr).min()
            xmax = (x + xerr).max()
        else:
            xmin = x.min()
            xmax = x.max()
        print(anova_df[[
            "group_name",
            "logit_group_es",
            "logit_es_std"
        ]])

        val = max(abs(xmin), abs(xmax)) * 1.1

        ax.set_ylim(len(anova_df['group_name']) - 1 + 0.5, -0.5)

        ax.tick_params('x', length=3, width=0.75)
        ax.tick_params('y', length=0)

        ax.spines['left'].set_linewidth(0.75)
        ax.spines['bottom'].set_linewidth(0.75)

        ax.axvline(0, ls='--', color='k', lw=0.5)
        ax.axvspan(-0.5, 0.5, color='grey', alpha=0.1, lw=0)

        ax.set_xlabel("Observed variant effect size (log2 ref/alt)")
        ax.set_ylabel("")
        ax.set_xlim(-val * 1.3, val * 1.3)
        ax.set_title(rsid)

        return ax

class PredictionImbalancePlotComponent(PlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, data, gs, fig):
        anova_df = data

        ax = fig.add_subplot(gs)

        x = anova_df["pred_logit_es"]
        xerr = anova_df["pred_logit_es_std"].fillna(0)
        
        # Error bars
        ax.errorbar(
            x=x,
            y=anova_df["group_name"],
            xerr=xerr,
            fmt='.',
            capsize=4,
            capthick=1.0,
            c='k',
            lw=1.0
        )

        # Points
        ax.scatter(
            y=anova_df["group_name"],
            x=x,
            lw=3,
            ec=anova_df["group_name"].map(self.names_to_colors).values,
            color='k',
            s=12
        )

        # formatting
        ax.set_yticklabels([
            label.get_text().replace(' (', '\n(').replace('/', '/\n')
            for label in ax.get_yticklabels()
        ])

        # horizontal separators
        for i in range(len(anova_df["group_name"]) - 1):
            ax.axhline(
                i + 0.5,
                ls='--',
                color='k',
                lw=0.5
            )

        xmin = (x - xerr).min()
        xmax = (x + xerr).max()

        val = max(abs(xmin), abs(xmax)) * 1.1

        ax.set_ylim(
            len(anova_df["group_name"]) - 1 + 0.5,
            -0.5
        )

        ax.tick_params('x', length=3, width=0.75)
        ax.tick_params('y', length=0)

        ax.spines['left'].set_linewidth(0.75)
        ax.spines['bottom'].set_linewidth(0.75)

        ax.axvline(0, ls='--', color='k', lw=0.5)
        ax.axvspan(
            -0.5,
            0.5,
            color='grey',
            alpha=0.1,
            lw=0
        )

        ax.set_xlabel(
            "Predicted variant effect size (log2 ref/alt)"
        )
        ax.set_ylabel("")

        ax.set_xlim(
            -val * 1.3,
            val * 1.3
        )

        ax.set_title("Model prediction")

        return ax


class DensityPlotComponent(PlotComponent):
    def __init__(self, common_norm=True, clip_max=None, clip_min=0, common_norm_quantile=1.0, **kwargs):
        super().__init__(**kwargs)
        self.common_norm = common_norm # wether to set a common y scale for all density plots
        self.common_norm_quantile = common_norm_quantile # quantile of max y values across rows to use as ylim
        self.clip_max = clip_max # hard threshold that ylim can not exceed
        self.clip_min = clip_min # hard threshold that ylim can not be lower than
        

    def plot(self, data, gs, fig):
        anova_df, dens, interval = data
        ref_allele = anova_df['ref'].values[0]
        alt_allele = anova_df['alt'].values[0]
        gs_subplot = gridspec.GridSpecFromSubplotSpec(len(anova_df), 1, gs, hspace=0)
        axes = []

        for i, group_name in enumerate(anova_df['group_name']):
            ax = fig.add_subplot(gs_subplot[i, :])
            signal_plot(interval, dens[group_name], color=self.names_to_colors[group_name], ax=ax)
            
            if i == len(anova_df)-1:
                ax.xaxis.set_visible(True)
                ax.set_xlabel(f"{anova_df['#chr'].values[0]}")
                ax.tick_params('x', length=3, width=0.7, which='major')
            else:
                ax.xaxis.set_visible(False)

            max_dens_value = np.nanmax(dens[group_name])
        
            if i == 0:
                ax.annotate(xy=((interval.end+interval.start)//2, max_dens_value), ha='center', text=f"{ref_allele}/{alt_allele}", xytext=(0, 25), textcoords="offset points", arrowprops=dict(arrowstyle="->"))

            ax.set_yticks([])
            ax.spines['left'].set_linewidth(0.75)
            ax.spines['bottom'].set_linewidth(0.75)
            pos = ax.get_position()
            ax.set_ylim(0, np.clip(max_dens_value, self.clip_min, self.clip_max))
            ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.8])
            axes.append(ax)

        if self.common_norm:
            top = np.nanquantile([np.nanmax(d) for d in dens.values()], self.common_norm_quantile)
            top = np.clip(top, self.clip_min, self.clip_max)
            for ax in axes:
                ax.set_ylim(0, top)
        
        common_ax = fig.add_subplot(gs)
        common_ax.axvline(0.5, ls='--', color='k', lw=0.5)
        common_ax.axis('off')
        common_ax.annotate(xy=(-0.1, 0.5), ha='center', va='center', rotation=90, text='DNase I density', annotation_clip=False)
        return axes, common_ax
        
class PredictionPlotComponent(PlotComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, data, gs, fig):
        anova_df = data

        gs_subplot = gridspec.GridSpecFromSubplotSpec(
            len(anova_df), 
            1, 
            subplot_spec=gs
        )

        axes = []

        for i, (_, row) in enumerate(anova_df.iterrows()):

            group_ax = fig.add_subplot(gs_subplot[i, :])
            axes.append(group_ax)

            # single prediction value
            group_ax.bar(
                x=[0],
                height=[row["pred_logit_es"]],
                color="grey"
            )

            group_ax.set_ylim(0, 1)

            group_ax.spines['left'].set_linewidth(0.75)
            group_ax.spines['bottom'].set_linewidth(0.75)

            group_ax.tick_params(
                'x',
                length=3,
                width=0.75,
                which='major'
            )

            group_ax.tick_params(
                'y',
                length=3,
                width=0.75,
                which='major'
            )

            group_ax.set_xticks([])

            if i == 0:
                group_ax.set_title("Prediction")
            else:
                group_ax.set_yticklabels([])


        axes[-1].set_xticks([0])
        axes[-1].set_xticklabels(["Geno"])

        return axes

class ContributionPlotComponent(PlotComponent):
    def __init__(self,
                 window=10, # display contributions from center - window to center + window
                 sequence_length=1344,
                 prediction_for_allele_by='prediction', # criterion to choose the allele to display contributions for
                 **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.sequence_length = sequence_length
        assert prediction_for_allele_by in ('prediction', 'effect', 0, 1, 2)
        self.prediction_for_allele_by = prediction_for_allele_by

    def plot(self, data, gs, fig):
        anova_df, shap_scores = data
        all_windows = []
        #normalize all same
        center = self.sequence_length // 2
        w1 = center - self.window
        w2 = center + self.window + 1
        
        for _, row in anova_df.iterrows():
            group = row["group_name"]
        
            ref_scores = shap_scores[group]["ref"]
            alt_scores = shap_scores[group]["alt"]
        
            z = alt_scores - ref_scores
        
            all_windows.append(z[w1:w2, :])
        
        all_windows = np.concatenate(all_windows, axis=0)
        
        global_abs = np.max(np.abs(all_windows))
        
        gs_subplot = gridspec.GridSpecFromSubplotSpec(len(anova_df), 1, subplot_spec=gs)
        hs_pos = [0]
        hs_neg = [0]
        axes = []
        for i, row in anova_df.iterrows():
            ax = fig.add_subplot(gs_subplot[i,:])
            center = self.sequence_length // 2
            w1 = center - self.window
            w2 = center + self.window + 1
        
            group = row["group_name"]
            scores = shap_scores[group]
            ref_scores = scores["ref"]
            alt_scores = scores["alt"]

            # combine alleles
            # difference between alleles
            z = alt_scores - ref_scores
            # only plot center window
            window = z[w1:w2, :].copy()
            # normalize for visualization
            # scale = np.max(np.abs(window))
            # if scale > 0:
            #     window = window / scale

            # plot sequence logo
            seq_plot(
                window,
                ax=ax
            )
            ax.set_ylim(-global_abs, global_abs)
            
            #add y axis label and ticks
            ax.axhline(0, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(
                axis="x",
                bottom=False,
                labelbottom=False
            )
            ax.set_ylabel("SHAP")
            if i == 0:
                ax.set_title("Contributions (SHAP)")
            axes.append(ax)

        # background axis for shared x coordinates
        big_ax = fig.add_subplot(gs)
        big_ax.set_xlim(
            0,
            2*self.window + 1
        )
        big_ax.axvspan(
            self.window - 1,
            self.window,
            color="grey",
            alpha=0.2,
            lw=0,
        )
        big_ax.set_axis_off()
        
        return axes

    