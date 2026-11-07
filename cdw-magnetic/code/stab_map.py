"""Stability map of the homogeneous ground state in (mu, B) (note stability/06).

At every node: certified homogeneous minimum M0 (magcdw.hom); for M0 > 0 the Goldstone and
radial kernels, for M0 = 0 the symmetric kernel (magcdw.stab.fast), scanned over q.

Usage (from the repository root):
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/stab_map.py --T 0.01 --n 60
    MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/stab_map.py --T 0.01 --plot-only
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from magcdw.hom.minimize import find_minimum  # noqa: E402
from magcdw.stab import fast  # noqa: E402

MU_MAX, B_MAX = 2.0, 7.0
QS = np.linspace(0.05, 3.0, 60)


def node(args):
    mu, B, T = args
    hm = find_minimum(B, mu, T)
    M0 = hm.mass
    if M0 > 1e-4:
        gold, rad = zip(*(fast.modes(q, M0, B, mu, T) for q in QS))
        gold, rad = np.array(gold), np.array(rad)
        ig, ir = int(np.argmin(gold)), int(np.argmin(rad))
        return {"mu": mu, "B": B, "M0": M0, "phase": hm.phase,
                "gold_min": float(gold[ig]), "q_gold": float(QS[ig]),
                "rad_min": float(rad[ir]), "q_rad": float(QS[ir])}
    sym = np.array([fast.symmetric_kernel(q, B, mu, T, regulator_mass=0.005 * math.sqrt(B)) for q in QS])
    i = int(np.argmin(sym))
    return {"mu": mu, "B": B, "M0": 0.0, "phase": hm.phase,
            "sym_min": float(sym[i]), "q_sym": float(QS[i])}


def compute(T, n, workers):
    mus = MU_MAX * np.arange(1, n + 1) / n
    fields = B_MAX * np.arange(1, n + 1) / n
    jobs = [(float(mu), float(B), T) for B in fields for mu in mus]
    start = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        nodes = list(pool.map(node, jobs, chunksize=8))
    return {"T": T, "g": "-pi", "mu": mus.tolist(), "B": fields.tolist(), "q_grid": QS.tolist(),
            "elapsed_s": time.perf_counter() - start, "nodes": nodes}


def plot(data, path, cdw=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mus, fields = np.array(data["mu"]), np.array(data["B"])
    n_mu, n_B = len(mus), len(fields)
    nodes = data["nodes"]
    least = np.full((n_mu, n_B), np.nan)       # min over modes and q
    which = np.zeros((n_mu, n_B))              # 0 symmetric, 1 Goldstone, 2 radial
    for k, nd in enumerate(nodes):
        j, i = divmod(k, n_mu)                 # jobs ordered B-major
        if nd["M0"] > 0:
            g, r = nd["gold_min"], nd["rad_min"]
            least[i, j], which[i, j] = (g, 1) if g <= r else (r, 2)
        else:
            least[i, j], which[i, j] = nd["sym_min"], 0
    extent = (0, fields[-1], 0, mus[-1])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    ax = axes[0]
    lim = np.nanmax(np.abs(least))
    scale = max(1e-6, np.nanpercentile(np.abs(least), 90))
    im = ax.imshow(np.clip(least, -scale, scale), origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu", norm=TwoSlopeNorm(0, -scale, scale), interpolation="nearest")
    fig.colorbar(im, ax=ax, label=r"$\min_q$ of the softest mode (clipped)")
    ax.contour(np.linspace(0, fields[-1], n_B), mus, least, levels=[0], colors="k", linewidths=0.8)
    ax.set_title(f"Stability of the homogeneous minimum, T = {data['T']} (red: unstable)")
    ax = axes[1]
    unstable = least < 0
    show = np.where(unstable, which, np.nan)
    ax.imshow(np.where(np.isnan(least), np.nan, 0 * least), origin="lower", aspect="auto",
              extent=extent, cmap="Greys", vmin=0, vmax=4)
    from matplotlib.colors import ListedColormap
    ax.imshow(show, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
              cmap=ListedColormap(["#8e44ad", "#c0392b", "#e67e22"]), vmin=-0.5, vmax=2.5)
    for c, name in zip(["#8e44ad", "#c0392b", "#e67e22"], ["unstable, M=0", "unstable Goldstone", "unstable radial"]):
        ax.scatter([], [], marker="s", color=c, label=name)
    if cdw:
        ax.scatter([n["B"] for n in cdw], [n["mu"] for n in cdw], s=4, color="k", label="CDW nodes (branch A, T=0.01)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Unstable mode type vs CDW map of branch A")
    for ax in axes:
        ax.set_xlabel(r"$|eB|/m_*^2$")
        ax.set_ylabel(r"$\mu/m_*$")
        ax.set_xlim(0, fields[-1])
        ax.set_ylim(0, mus[-1])
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=0.01)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    out = HERE / f"stab-map-T{args.T:g}.json"
    if not args.plot_only:
        data = compute(args.T, args.n, args.workers)
        out.write_text(json.dumps(data))
        print(f"T={args.T}: {args.n}x{args.n} nodes in {data['elapsed_s']:.0f}s")
    data = json.loads(out.read_text())
    cdw = None
    amap = HERE / "adaptive-map.json"
    if amap.exists():
        cdw = [n for n in json.loads(amap.read_text())["nodes"] if n["label"] == "cdw"]
    plot(data, HERE.parent / "figures" / f"stab-map-T{args.T:g}.png", cdw)


if __name__ == "__main__":
    main()
