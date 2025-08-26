from matplotlib.ticker import FuncFormatter, LinearLocator
import numpy as np
import matplotlib.pyplot as plt

def visualize_fric_sensitivity(
    df,
    series_left,
    left_label,
    left_units,
    series_right=None,
    right_label=None,
    right_units=None,
    separate_axes=False,
    title="Friction Sensitivity",
    figsize=(3, 4.25),
    dpi=300,
    cmap_name="Blues"
):
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11})

    x = df["Friction Gradient"].values
    left_colors = plt.get_cmap(cmap_name)(np.linspace(0.60, 0.90, len(series_left)))

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

    handles, labels = [], []
    for i, col in enumerate(series_left):
        h, = ax1.plot(x, df[col], 'o-', color=left_colors[i], markersize=3, linewidth=1)
        handles.append(h); labels.append(col)

    ax1.set_xlabel('Frictional Gradient [MPa/m]')
    ax1.set_ylabel(f"{left_label} {left_units}")

    def sci3(val, pos):
        if val == 0: return "0.00e00"
        m, e = f"{val:.2e}".split('e')
        e = int(e)
        return f"{float(m):.2f}e{'-' if e < 0 else ''}{abs(e):02d}"
    ax1.xaxis.set_major_formatter(FuncFormatter(sci3))
    ax1.yaxis.set_major_formatter(FuncFormatter(sci3))

    ax1.set_xlim(0.0, 0.0010)
    ax1.set_xticks(np.linspace(0.0, 0.0010, 6))
    ax1.tick_params(axis='x', labelrotation=90, pad=2)

    # ----- CHANGED BLOCK: add top/bottom buffer but keep 10 y labels -----
    yL = np.hstack([df[col].values for col in series_left])
    y_min, y_max = float(np.min(yL)), float(np.max(yL))
    if y_min == y_max:
        # Degenerate range: pad symmetrically around the single value
        pad = (abs(y_max) if y_max != 0 else 1.0) * 0.06
        lo, hi = y_min - pad, y_max + pad
    else:
        rng = y_max - y_min
        pad = 0.06 * rng                              
        lo, hi = y_min - pad, y_max + pad

    ax1.set_ylim(lo, hi)                              
    ax1.yaxis.set_major_locator(LinearLocator(10))    

    ax1.grid(False, axis='x')
    ax1.grid(True, axis='y', which='major', linestyle='--', alpha=0.3)

    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_ticks_position('left')
    ax1.tick_params(axis='y', which='both', right=False, labelright=False)
    ax1.spines['left'].set_color('black')
    ax1.yaxis.label.set_color('black')
    ax1.tick_params(axis='y', colors='black')

    legend = ax1.legend(
        handles, labels,
        ncol=2, frameon=False,
        loc='upper center', bbox_to_anchor=(0.5, -0.28),
        fontsize=11, title_fontsize=11, borderaxespad=0.0
    )
    for idx, text in enumerate(legend.get_texts()):
        text.set_color(handles[idx].get_color())

    fig.suptitle("")
    plt.tight_layout(rect=[0.06, 0.4, 0.98, 0.98])
    return fig, ax1
