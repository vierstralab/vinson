import logging

import matplotlib.pyplot as plt
from typing import Sequence, Union
from matplotlib import rcParams
from matplotlib import gridspec

class LoggerMixin:
    def __init__(self, logger_level: int = logging.INFO):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logger_level)


class PlotComponent(LoggerMixin):
    def __init__(
        self,
        width: float = 1.0,
        margins: Union[float, Sequence[float]] = 0.1,
        names_to_colors: dict = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.width = width
        self.names_to_colors = names_to_colors

        if isinstance(margins, Sequence) and not isinstance(margins, str):
            if len(margins) != 2:
                raise ValueError("margins must be length 2")
            self.margin_left, self.margin_right = margins
        else:
            self.margin_left = self.margin_right = margins

    def plot(self, data, gs, fig, **kwargs):
        raise NotImplementedError
        
def setup_gridspec(plot_components, fig=None):

    width_ratios = [
        x
        for c in plot_components
        for x in [
            c.margin_left,
            c.width,
            c.margin_right
        ]
    ]

    return gridspec.GridSpec(
        1,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        wspace=0,
        left=0.07,
        right=0.98,
        bottom=0.18,
        top=0.83,
    )
    
def plot_variant(
    data,
    fig=None,
    group_height=1.1,
    inches_per_unit=0.75,
    save_path=None,
):
    plot_components, variant_data, num_groups = data

    if fig is None:
        total_width = sum(
            c.margin_left + c.width + c.margin_right
            for c in plot_components
        )

        fig = plt.figure(
            figsize=(
                max(12, total_width * inches_per_unit),
                max(3.5, num_groups * group_height)
            )
        )

    component_axes = []

    master_gs = setup_gridspec(plot_components, fig=fig)

    for i, (component, component_data) in enumerate(
        zip(plot_components, variant_data)
    ):
        component_axes.append(
            component.plot(
                component_data,
                gs=master_gs[:, 3*i + 1],
                fig=fig
            )
        )

    # Optional PDF export
    if save_path is not None:
        fig.savefig(
            save_path,
            format="pdf",
            bbox_inches="tight",
        )

    return component_axes, master_gs