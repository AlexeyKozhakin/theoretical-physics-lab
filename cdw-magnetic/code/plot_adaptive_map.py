"""Plot adaptive-map.json (note 18): phases, M and b of the minimum, CDW gain.

Axes: x = |eB|, y = mu. Every pixel takes the value of the nearest computed node
(in lattice units, so the mu and B steps weigh equally); pixel edges are not
resolved phase boundaries.

Usage (from the repository root):
    MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/plot_adaptive_map.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
COLORS = {"sym": "#e8e4d8", "hom": "#4a78b5", "cdw": "#c8372d"}
ORDER = ["sym", "hom", "cdw"]
NAMES = {"sym": "symmetric, M=0", "hom": "homogeneous, M>0", "cdw": "CDW, M>0, b>0"}


def minimum_mass(node):
    if node["label"] == "cdw":
        return node["mass"]
    return node["hom_mass"] if node["label"] == "hom" else 0.0


def minimum_wave(node):
    return node["wave"] if node["label"] == "cdw" else 0.0


def raster(data, nodes, values):
    """Nearest-node image with rows = mu (y) and columns = B (x)."""
    ij = np.array([[n["i"], n["j"]] for n in nodes], float)
    unit = data["coarse"] * 2 ** data["levels"]
    gi, gj = np.meshgrid(np.arange(1, unit + 1), np.arange(1, unit + 1), indexing="ij")
    _, idx = cKDTree(ij).query(np.c_[gi.ravel(), gj.ravel()])
    return np.asarray(values)[idx].reshape(gi.shape)        # [mu index, B index]


def axes_style(ax, data, title):
    ax.set_xlabel(r"$|eB|/m_*^2$")
    ax.set_ylabel(r"$\mu/m_*$")
    ax.set_xlim(0, data["B_max"])
    ax.set_ylim(0, data["mu_max"])
    ax.set_title(title)


def main():
    data = json.loads((HERE / "adaptive-map.json").read_text())
    nodes = data["nodes"]
    extent = (0, data["B_max"], 0, data["mu_max"])

    labels = raster(data, nodes, [ORDER.index(n["label"]) for n in nodes])
    masses = raster(data, nodes, [minimum_mass(n) for n in nodes])
    waves = raster(data, nodes, [minimum_wave(n) for n in nodes])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.imshow(labels, origin="lower", aspect="auto", interpolation="nearest", extent=extent,
              cmap=ListedColormap([COLORS[k] for k in ORDER]), vmin=-0.5, vmax=2.5)
    for k in ORDER:
        ax.scatter([], [], marker="s", color=COLORS[k], label=NAMES[k])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes_style(ax, data, f"Phase, T = {data['T']} ({len(nodes)} nodes)")

    ax = axes[1]
    im = ax.imshow(masses, origin="lower", aspect="auto", interpolation="nearest",
                   extent=extent, cmap="viridis", vmin=0)
    fig.colorbar(im, ax=ax, label=r"$M/m_*$ at the minimum")
    axes_style(ax, data, r"$M$ of the global minimum")

    ax = axes[2]
    shown = np.ma.masked_where(labels != ORDER.index("cdw"), waves)
    ax.imshow(np.where(labels == ORDER.index("cdw"), np.nan, labels), origin="lower",
              aspect="auto", interpolation="nearest", extent=extent,
              cmap=ListedColormap(["#f2f0ea", "#dfe6f0", "#f2f0ea"]), vmin=-0.5, vmax=2.5)
    im = ax.imshow(shown, origin="lower", aspect="auto", interpolation="nearest",
                   extent=extent, cmap="plasma")
    fig.colorbar(im, ax=ax, label=r"$b/m_*$ at the minimum (CDW only)")
    axes_style(ax, data, r"$b$ of the global minimum ($b=0$ elsewhere)")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "adaptive-map.png", dpi=150)
    plt.close(fig)

    # Computed nodes and CDW gain
    cdw = [n for n in nodes if n["label"] == "cdw"]
    other = [n for n in nodes if n["label"] != "cdw"]
    fig, ax = plt.subplots(figsize=(7, 5.2), constrained_layout=True)
    ax.scatter([n["B"] for n in other], [n["mu"] for n in other], s=1, color="#cccccc")
    if cdw:
        gain = np.array([-max(n["direct_gaps"]) for n in cdw])
        sc = ax.scatter([n["B"] for n in cdw], [n["mu"] for n in cdw], c=gain, s=8,
                        cmap="magma_r", norm=LogNorm(vmin=max(gain.min(), 1e-8), vmax=gain.max()))
        fig.colorbar(sc, ax=ax, label=r"CDW gain $-\delta$ ($m_*^3$)")
    axes_style(ax, data, "Computed nodes and CDW gain")
    fig.savefig(FIG / "adaptive-map-gain.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
