import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional, Union, Dict, Any
from applycommonstylequantfun import _apply_common_style_quant

def quant_vis(
    x: Sequence[float],
    y: Sequence[float],
    ax: Optional[plt.Axes] = None,
    *,
    kind: str = 'line',         
    yerr: Optional[Sequence[float]] = None,
    xerr: Optional[Sequence[float]] = None,       
    xlabel: str = '',
    ylabel: str = '',
    title: str = '',
    figsize: tuple = (5, 5),
    dpi: int = 350,
    color: Union[str, Sequence[str]] = None,
    palette: str = 'hsv',
    linestyle: str = '-',
    marker: Optional[str] = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    grid: bool = True,
    logx: bool = False,
    logy: bool = False,
    annotate: bool = False,
    annotate_fmt: str = '{:.2f}',
    annotate_offset: tuple = (3, 3),
    fill_between: bool = False,
    fill_kwargs: Dict[str, Any] = None,
    **plot_kwargs
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        ax.figure.set_dpi(dpi)
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    if grid: ax.grid(True, which='both', linestyle='--', alpha=0.3, color='grey')

    cmap = plt.get_cmap(palette)
    col  = color if color is not None else cmap(0.0)

    if kind == 'errorbar':
        ax.errorbar(
            x, y,
            yerr=yerr,
            xerr=xerr,
            color=col,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            alpha=alpha,
            elinewidth=1.2,
            capsize=3,
            **plot_kwargs
        )
    elif kind == 'scatter':
        ax.scatter(x, y, c=col, marker=marker or 'o', alpha=alpha, **plot_kwargs)
    else:
        ax.plot(
            x, y,
            color=col,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            alpha=alpha,
            **plot_kwargs
        )
        if fill_between and yerr is not None:
            fb = dict(alpha=0.2, color=col)
            if fill_kwargs: fb.update(fill_kwargs)
            arr_y = np.array(y); err = np.array(yerr)
            ax.fill_between(x, arr_y-err, arr_y+err, **fb)

    if annotate:
        for xi, yi in zip(x, y):
            ax.annotate(
                annotate_fmt.format(yi),
                (xi, yi),
                textcoords='offset points',
                xytext=annotate_offset,
                fontsize=8,
                color='black'
            )

    _apply_common_style_quant(ax, xlabel, ylabel, title)
    return ax
