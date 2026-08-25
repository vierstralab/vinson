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
        
def setup_gridspec(plot_components):
    """
    Setup the gridspec for the plots.
    Components are plotted in the middle column, with margins on the left and right.
    """
    width_ratios = [
        x
        for c in plot_components
        for x in [c.margin_left, c.width, c.margin_right]
    ]
    return gridspec.GridSpec(1, len(width_ratios), width_ratios=width_ratios, wspace=0, hspace=0)
    
def plot_variant(data,
                 fig=None,
                 group_height=0.4,
                 inches_per_unit=1.0
                ):
    plot_components, variant_data, num_groups = data
    
    if fig is None:
        fig = plt.figure(figsize=(
            sum(x for c in plot_components for x in [c.margin_left, c.width, c.margin_right]) * inches_per_unit,
            num_groups * group_height * inches_per_unit))
    
    component_axes = []

    master_gs = setup_gridspec(plot_components)
    for i, (component, component_data) in enumerate(zip(plot_components, variant_data)):
        component_axes.append(component.plot(component_data, gs=master_gs[:, 3*i + 1], fig=fig))
    
    return component_axes, master_gs