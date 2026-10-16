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

    def __init__(self, use_stripplot=True, **kwargs):
        super().__init__(**kwargs)
        self.use_stripplot = use_stripplot

    def plot(self, data, gs, fig):
        tested_df, anova_df = data

        ref_allele = anova_df["ref"].iloc[0]
        alt_allele = anova_df["alt"].iloc[0]
        rsid = anova_df["variant_id"].iloc[0]

        ax = fig.add_subplot(gs)

        groups = anova_df["group_name"].tolist()
        y = np.arange(len(groups))

        # ---------------------------------------
        # 1. Individual sample observations
        # ---------------------------------------
        if self.use_stripplot:
            sns.stripplot(
                data=tested_df,
                y="group_name",
                x="logit_es",
                order=groups,
                ax=ax,
                size=5,
                jitter=0.1,
                orient="h",
                hue="group_name",
                hue_order=groups,
                palette=self.names_to_colors,
                alpha=0.3,
                linewidth=0,
                legend=False,
                zorder=1,
            )

        # ---------------------------------------
        # 2. Group mean and error bars
        # ---------------------------------------
        x = anova_df["logit_group_es"].to_numpy()
        xerr = anova_df["logit_es_std"].fillna(0).to_numpy()

        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            fmt="none",
            ecolor="black",
            elinewidth=2,
            capsize=5,
            capthick=2,
            zorder=4,
        )

        ax.scatter(
            x=x,
            y=y,
            s=85,
            color="black",
            edgecolors=[
                self.names_to_colors[g] for g in groups
            ],
            linewidths=2,
            zorder=5,
        )

        # ---------------------------------------
        # 3. Zero line and shaded region
        # ---------------------------------------
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        ax.axvspan(
            -0.5,
            0.5,
            color="grey",
            alpha=0.10,
            lw=0,
        )

        # ---------------------------------------
        # 4. Horizontal separators
        # ---------------------------------------
        for i in range(len(groups) - 1):
            ax.axhline(
                i + 0.5,
                color="black",
                linestyle="--",
                linewidth=0.7,
            )

        # ---------------------------------------
        # 5. Axis formatting
        # ---------------------------------------
        ax.set_ylim(len(groups) - 0.5, -0.5)

        # Hide cell names here (display once elsewhere)
        # ax.set_yticks(y)
        # ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

        ax.set_xlim(-4, 4)
        ax.set_xticks([-4, -2, 0, 2, 4])

        ax.set_xlabel(
            "Variant effect size (log2 ref/alt)",
            fontsize=11,
        )
        ax.set_ylabel("")

        ax.tick_params(axis="x", labelsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---------------------------------------
        # 6. Allele labels
        # ---------------------------------------
        trans = transforms.blended_transform_factory(
            ax.transAxes,
            ax.transAxes
        )

        ax.annotate(
            alt_allele,
            xy=(0.1, 1.08),
            xycoords=trans,
            fontname="IBM Plex Mono",
            fontweight="bold",
            fontsize=11,
            ha="center",
            bbox=dict(
                boxstyle="circle",
                ec="none",
                fc=cm.map_vocab_color(alt_allele, "dna")[0],
            ),
            annotation_clip=False,
        )

        ax.annotate(
            ref_allele,
            xy=(0.9, 1.08),
            xycoords=trans,
            fontname="IBM Plex Mono",
            fontweight="bold",
            fontsize=11,
            ha="center",
            bbox=dict(
                boxstyle="circle",
                ec="none",
                fc=cm.map_vocab_color(ref_allele, "dna")[0],
            ),
            annotation_clip=False,
        )

        ax.set_title(
            rsid,
            fontsize=13,
            pad=16,
        )
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False, labelleft=False)

        return ax

class PredictionImbalancePlotComponent(PlotComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, data, gs, fig):

        anova_df = data

        ax = fig.add_subplot(gs)

        groups = anova_df["group_name"].tolist()
        y = np.arange(len(groups))

        # ---------------------------------------
        # 1. Predicted group effects and std
        # ---------------------------------------
        x = anova_df["pred_logit_es"].to_numpy()
        xerr = anova_df["pred_logit_es_std"].fillna(0).to_numpy()

        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            fmt="none",
            ecolor="black",
            elinewidth=2,
            capsize=5,
            capthick=2,
            zorder=4,
        )

        ax.scatter(
            x=x,
            y=y,
            s=85,
            color="black",
            edgecolors=[
                self.names_to_colors[g]
                for g in groups
            ],
            linewidths=2,
            zorder=5,
        )

        # ---------------------------------------
        # 2. Zero line and shaded region
        # ---------------------------------------
        ax.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=1,
        )

        ax.axvspan(
            -0.5,
            0.5,
            color="grey",
            alpha=0.10,
            lw=0,
        )

        # ---------------------------------------
        # 3. Horizontal separators
        # ---------------------------------------
        for i in range(len(groups) - 1):
            ax.axhline(
                i + 0.5,
                color="black",
                linestyle="--",
                linewidth=0.7,
            )

        # ---------------------------------------
        # 4. Axis formatting
        # ---------------------------------------
        ax.set_ylim(len(groups) - 0.5, -0.5)

        # No cell-type labels here
        # ax.set_yticks(y)
        # ax.set_yticklabels([])
        # ax.tick_params(axis="y", length=0)

        # Same limits as observed imbalance
        ax.set_xlim(-4, 4)
        ax.set_xticks([-4, -2, 0, 2, 4])

        ax.set_xlabel(
            "Predicted variant effect size (log2 ref/alt)",
            fontsize=11,
        )

        ax.set_ylabel("")

        ax.tick_params(axis="x", labelsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_title(
            "Model prediction",
            fontsize=13,
            pad=16,
        )

        return ax


class DensityPlotComponent(PlotComponent):

    def __init__(
        self,
        common_norm=True,
        clip_max=None,
        clip_min=0,
        common_norm_quantile=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.common_norm = common_norm
        self.common_norm_quantile = common_norm_quantile
        self.clip_max = clip_max
        self.clip_min = clip_min

    def plot(self, data, gs, fig):

        anova_df, dens, interval = data

        ref_allele = anova_df["ref"].iloc[0]
        alt_allele = anova_df["alt"].iloc[0]

        groups = anova_df["group_name"].tolist()

        # One density plot per group
        gs_subplot = gridspec.GridSpecFromSubplotSpec(
            len(groups),
            1,
            subplot_spec=gs,
            hspace=0.15,
        )

        axes = []

        # ---------------------------------------
        # 1. Plot density for each group
        # ---------------------------------------
        for i, group_name in enumerate(groups):

            ax = fig.add_subplot(gs_subplot[i, :])

            signal_plot(
                interval,
                dens[group_name],
                color=self.names_to_colors[group_name],
                ax=ax,
            )

            # Variant position
            variant_pos = (
                interval.start + interval.end
            ) / 2

            ax.axvline(
                variant_pos,
                color="black",
                linestyle="--",
                linewidth=0.8,
            )

            # Show x-axis only on bottom group
            if i == len(groups) - 1:
                ax.xaxis.set_visible(True)
                ax.set_xlabel(
                    anova_df["#chr"].iloc[0],
                    fontsize=11,
                )
                ax.tick_params(
                    axis="x",
                    length=3,
                    width=0.7,
                    labelsize=9,
                )
            else:
                ax.xaxis.set_visible(False)

            max_dens_value = np.nanmax(
                dens[group_name]
            )

            ax.set_yticks([])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.spines["left"].set_linewidth(0.75)
            ax.spines["bottom"].set_linewidth(0.75)

            ax.set_ylim(
                0,
                np.clip(
                    max_dens_value,
                    self.clip_min,
                    self.clip_max,
                )
            )

            axes.append(ax)

        # ---------------------------------------
        # 2. Common normalization
        # ---------------------------------------
        if self.common_norm:

            top = np.nanquantile(
                [
                    np.nanmax(dens[g])
                    for g in groups
                ],
                self.common_norm_quantile,
            )

            top = np.clip(
                top,
                self.clip_min,
                self.clip_max,
            )

            for ax in axes:
                ax.set_ylim(0, top)

        # ---------------------------------------
        # 3. Cell-type labels
        #    Display once, to right of density
        # ---------------------------------------
        for i, group_name in enumerate(groups):
            label = group_name.replace("_", " ")
        
            axes[i].text(
                1.12,
                0.5,
                label,
                transform=axes[i].transAxes,
                ha="left",
                va="center",
                fontsize=11,
                clip_on=False,
            )

            

        # ---------------------------------------
        # 4. Allele annotation above density
        # ---------------------------------------
        axes[0].annotate(
            f"{ref_allele}/{alt_allele}",
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1,
            ),
            annotation_clip=False,
        )

        # ---------------------------------------
        # 5. Shared DNase density label
        # ---------------------------------------
        common_ax = fig.add_subplot(gs)

        common_ax.axis("off")

        common_ax.text(
            -0.12,
            0.5,
            "DNase I density",
            transform=common_ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            clip_on=False,
        )

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
        
        gs_subplot = gridspec.GridSpecFromSubplotSpec(
            len(anova_df),
            1,
            subplot_spec=gs,
            hspace=0.15,
        )
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
            ax.set_ylim(-global_abs * 1.1, global_abs * 1.1)
            
            #add y axis label and ticks
            ax.axhline(0, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(
                axis="x",
                bottom=False,
                labelbottom=False
            )
            ax.set_ylabel("")
            ax.set_yticks([])
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

    