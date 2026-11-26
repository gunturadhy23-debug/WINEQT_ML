"""Helper grafik Matplotlib dengan gaya yang seragam."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BAIK = "#8B1E3F"     # anggur merah
BIASA = "#6F93C0"    # biru lembut
EMAS = "#C9A24B"
HIJAU = "#5E8C6A"
INK = "#2B2124"
MUTE = "#8A7F83"
GRID = "#E6DDE0"
PALET = [BAIK, BIASA, EMAS, HIJAU, "#7B6D8D"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": GRID, "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "axes.titlecolor": INK,
})


def gaya(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return ax


def fig_baru(w=6.0, h=3.4, grid="y"):
    fig, ax = plt.subplots(figsize=(w, h))
    gaya(ax, grid)
    return fig, ax


def label_batang(ax, bars, fmt="{:,.0f}", dy=0.01):
    top = ax.get_ylim()[1]
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8, color=INK)


def donat(ax, nilai, label, warna, judul=None):
    wedges, _ = ax.pie(nilai, colors=warna, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    total = sum(nilai)
    for w, n, l in zip(wedges, nilai, label):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        ax.text(0.81 * np.cos(ang), 0.81 * np.sin(ang), f"{n / total * 100:.1f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.legend(wedges, label, loc="center", frameon=False, fontsize=8)
    if judul:
        ax.set_title(judul)
    ax.set_aspect("equal")


def heatmap(ax, M, xlab, ylab, fmt="{:.2f}", cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fs=7):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlab)))
    ax.set_xticklabels(xlab, rotation=45, ha="right")
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab)
    for s in ax.spines.values():
        s.set_visible(False)
    if annot:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                        color="white" if abs(v) > 0.55 * max(abs(vmin), abs(vmax)) else INK)
    return im
