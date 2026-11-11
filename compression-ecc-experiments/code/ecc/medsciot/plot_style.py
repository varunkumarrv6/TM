"""Shared plotting helpers: grouped bars with exact value labels, an exact-value table under every panel,
and a labelled benchmark line."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
N_TEST = 105_066                     # MedSec-25 test flows: accuracy = correct flows / N_TEST
# fixed left-to-right order and colour for every protection scheme
SCHEMES = [("none", "No protection", "#eb6834"),
           ("SEQ", "Hamming – SEQ", "#eda100"),
           ("INT", "Hamming – INT", "#1baf7a"),
           ("KEY", "Hamming – KEY", "#2a78d6")]


def setup():
    plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
                         "axes.axisbelow": True, "axes.titlesize": 12.5, "axes.titlelocation": "left"})


def _text_color(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "white"


def _delta(v, bench, decimals):
    return round(v - bench, decimals) + 0.0          # + 0.0 turns -0.0 into 0.0


def grouped_bars(ax, categories, series, bench=None, ylim=(0, 118), decimals=4, table=True, n_total=N_TEST):
    """series: list of (label, values in %, colour), drawn left to right inside each category.
    Each bar shows its exact accuracy (and change vs benchmark); with table=True an aligned table under the panel
    gives accuracy, change vs benchmark and the exact number of correctly classified test flows."""
    x = np.arange(len(categories))
    n = len(series)
    width = 0.84 / n
    for i, (label, values, color) in enumerate(series):
        xs = x + (i - (n - 1) / 2) * width
        ax.bar(xs, values, width * 0.92, color=color, label=label, zorder=3, edgecolor="white", linewidth=0.6)
        for xc, v in zip(xs, values):
            txt = f"{v:.{decimals}f}"
            if bench is not None:
                txt += f"  ({_delta(v, bench, decimals):+.{decimals}f})"
            if v >= 50:   # inside the bar, reading upwards from below the top
                ax.text(xc, v - 2, txt, rotation=90, ha="center", va="top", fontsize=8.5,
                        color=_text_color(color), fontweight="bold", zorder=4)
            else:         # short bar: label sits above it
                ax.text(xc, v + 2, txt, rotation=90, ha="center", va="bottom", fontsize=8.5, color=INK, zorder=4)
    if bench is not None:
        ax.axhline(bench, color=INK, ls="--", lw=1.3, zorder=5)
        tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.005, bench, f"benchmark\n{bench:.{decimals}f} %", transform=tr, ha="left", va="center",
                fontsize=9.5, color=INK, clip_on=False)
    ax.set_xticks(x, categories)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, len(categories) - 0.45)      # 1 data unit per category -> table columns line up
    ax.set_ylabel("accuracy (%)")
    if table:
        cells = []
        for label, values, color in series:
            row = []
            for v in values:
                correct = int(round(v / 100 * n_total))
                d = "" if bench is None else f"   Δ {_delta(v, bench, decimals):+.{decimals}f}"
                row.append(f"{v:.{decimals}f} %{d}\n{correct:,} / {n_total:,} correct")
            cells.append(row)
        ax_h_in = ax.get_position().height * ax.figure.get_figheight()
        tb_h, gap = ROW_H * n / ax_h_in, TICK_GAP / ax_h_in            # inches -> axes fraction
        tb = ax.table(cellText=cells, rowLabels=[s[0] for s in series], cellLoc="center", rowLoc="right",
                      bbox=[0, -gap - tb_h, 1, tb_h])
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.6)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(GRID)
            if c == -1:                                   # row label = series colour swatch
                cell.set_facecolor(series[r][2])
                cell.get_text().set_color(_text_color(series[r][2]))
                cell.get_text().set_fontweight("bold")
    return ax


ROW_H, TICK_GAP = 0.44, 0.8          # inches: one table row, room for the 2-line group labels above the table


def figure(n_panels, n_series, panel_h=4.6, width=14, title_h=1.15, between=0.6):
    """Figure with n_panels stacked panels, each followed by room for its exact-value table."""
    block = panel_h + TICK_GAP + ROW_H * n_series + between
    H = title_h + n_panels * block
    fig = plt.figure(figsize=(width, H))
    axes = []
    for i in range(n_panels):
        top_in = H - title_h - i * block
        axes.append(fig.add_axes([0.17, (top_in - panel_h) / H, 0.72, (panel_h - 0.45) / H]))
    return fig, axes


def top_legend(fig, ax, extra_note="value on each bar = exact accuracy %  (change vs benchmark)  ·  table = exact counts"):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.998), ncol=len(l), frameon=False,
               title="bars, left → right in every group:", title_fontsize=10, alignment="left")
    if extra_note:
        fig.text(0.99, 0.998, extra_note, ha="right", va="top", fontsize=9.5, color=INK2)
