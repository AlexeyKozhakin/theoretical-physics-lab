"""Overlay: inhomogeneous map (inh-adaptive-map.json) and stability map (stab-adaptive-map.json), note 20.

Usage (from the repository root):
    MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/plot_inh_vs_stab.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent


def raster(data, values):
    ij = np.array([[n["i"], n["j"]] for n in data["nodes"]], float)
    unit = data["coarse"] * 2 ** data["levels"]
    gi, gj = np.meshgrid(np.arange(1, unit + 1), np.arange(1, unit + 1), indexing="ij")
    _, idx = cKDTree(ij).query(np.c_[gi.ravel(), gj.ravel()])
    return np.asarray(values)[idx].reshape(gi.shape)


def main():
    inh = json.loads((HERE / "inh-adaptive-map.json").read_text())
    stab = json.loads((HERE / "stab-adaptive-map.json").read_text())
    order = ["sym", "hom", "cdw"]
    phase = raster(inh, [order.index(n["label"]) for n in inh["nodes"]])
    unstable = raster(stab, [0.0 if n["label"] == "S" else 1.0 for n in stab["nodes"]])
    extent = (inh["B_min"], inh["B_max"], inh["mu_min"], inh["mu_max"])
    fig, ax = plt.subplots(figsize=(7.5, 5.6), constrained_layout=True)
    ax.imshow(phase, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
              cmap=ListedColormap(["#efece4", "#9db8d9", "#f0a08f"]), vmin=-0.5, vmax=2.5)
    ax.imshow(np.ma.masked_where(unstable < 0.5, unstable), origin="lower", aspect="auto",
              extent=(stab["B_min"], stab["B_max"], stab["mu_min"], stab["mu_max"]),
              interpolation="nearest", cmap=ListedColormap(["#5b2c6f"]), alpha=0.75)
    for c, name in zip(["#efece4", "#9db8d9", "#f0a08f", "#5b2c6f"],
                       ["symmetric (branch A)", "homogeneous M>0 (branch A)", "CDW (branch A)",
                        "homogeneous state unstable (branch B)"]):
        ax.scatter([], [], marker="s", color=c, label=name)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_xlabel(r"$|eB|/m_*^2$")
    ax.set_ylabel(r"$\mu/m_*$")
    ax.set_title(f"CDW map vs local instability, T = {inh['T']}")
    fig.savefig(HERE.parent / "figures" / "inh-vs-stab.png", dpi=150)


if __name__ == "__main__":
    main()
