"""Plot adaptive-map.json (note 18): phases and CDW gain.

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


def main():
    data = json.loads((HERE / "adaptive-map.json").read_text())
    nodes = data["nodes"]
    mu = np.array([n["mu"] for n in nodes])
    B = np.array([n["B"] for n in nodes])
    lab = np.array([ORDER.index(n["label"]) for n in nodes])
    # Nearest computed node on a fine raster (in lattice units, so mu and B steps weigh equally).
    ij = np.array([[n["i"], n["j"]] for n in nodes], float)
    tree = cKDTree(ij)
    unit = data["coarse"] * 2 ** data["levels"]
    gi, gj = np.meshgrid(np.arange(1, unit + 1), np.arange(1, unit + 1))
    _, idx = tree.query(np.c_[gi.ravel(), gj.ravel()])
    image = lab[idx].reshape(gi.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.imshow(image, origin="lower", aspect="auto", interpolation="nearest",
              extent=(0, data["mu_max"], 0, data["B_max"]),
              cmap=ListedColormap([COLORS[k] for k in ORDER]), vmin=-0.5, vmax=2.5)
    edge = [n for n in nodes if n.get("at_wave_cap")]
    if edge:
        ax.scatter([n["mu"] for n in edge], [n["B"] for n in edge], marker="x", s=12,
                   color="k", lw=0.7, label="CDW at the b-range edge")
    for k, name in zip(ORDER, ["symmetric, M=0", "homogeneous, M>0", "CDW, M>0, b>0"]):
        ax.scatter([], [], marker="s", color=COLORS[k], label=name)
    ax.set_xlabel(r"$\mu/m_*$")
    ax.set_ylabel(r"$|eB|/m_*^2$")
    ax.set_title(f"Phases, T = {data['T']} (fast engine, {len(nodes)} nodes)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax = axes[1]
    cdw = [n for n in nodes if n["label"] == "cdw"]
    other = [n for n in nodes if n["label"] != "cdw"]
    ax.scatter([n["mu"] for n in other], [n["B"] for n in other], s=1, color="#cccccc")
    if cdw:
        gain = np.array([-max(n["direct_gaps"]) for n in cdw])
        sc = ax.scatter([n["mu"] for n in cdw], [n["B"] for n in cdw], c=gain, s=8,
                        cmap="magma_r", norm=LogNorm(vmin=max(gain.min(), 1e-8), vmax=gain.max()))
        fig.colorbar(sc, ax=ax, label=r"CDW gain $-\delta$ ($m_*^3$)")
    ax.set_xlim(0, data["mu_max"])
    ax.set_ylim(0, data["B_max"])
    ax.set_xlabel(r"$\mu/m_*$")
    ax.set_ylabel(r"$|eB|/m_*^2$")
    ax.set_title("Computed nodes and CDW gain")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "adaptive-map.png", dpi=150)

    # Condensate coordinates of CDW nodes
    if cdw:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
        for ax, key, name in ((axes[0], "mass", "M"), (axes[1], "wave", "b")):
            sc = ax.scatter([n["mu"] for n in cdw], [n["B"] for n in cdw],
                            c=[n[key] for n in cdw], s=8, cmap="viridis")
            fig.colorbar(sc, ax=ax, label=f"${name}/m_*$")
            ax.set_xlim(0, data["mu_max"])
            ax.set_ylim(0, data["B_max"])
            ax.set_xlabel(r"$\mu/m_*$")
            ax.set_ylabel(r"$|eB|/m_*^2$")
            ax.set_title(f"CDW nodes: {name}")
        fig.savefig(FIG / "adaptive-map-condensates.png", dpi=150)


if __name__ == "__main__":
    main()
