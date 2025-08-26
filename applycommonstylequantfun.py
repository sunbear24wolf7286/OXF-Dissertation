import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional, Union, Dict, Any

def _apply_common_style_quant(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    xpad: float = 0.05,
    ypad: float = 0.05
):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif']  = ['Times New Roman']
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    x0, x1 = ax.get_xlim(); dx = (x1 - x0) * xpad
    y0, y1 = ax.get_ylim(); dy = (y1 - y0) * ypad
    ax.set_xlim(x0 - dx, x1 + dx)
    ax.set_ylim(y0 - dy, y1 + dy)
    return ax