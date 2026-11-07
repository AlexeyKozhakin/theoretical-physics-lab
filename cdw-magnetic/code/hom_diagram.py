"""Homogeneous (mu, B) phase diagram with the certified minimum of magcdw.hom (stage D1/D2).

Usage (from the repository root):
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/hom_diagram.py --T 0 --n 240
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/hom_diagram.py --T 0 --n 300 --mu 0.75 1.5 --B 0 1
    MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/hom_diagram.py --plot-only --T 0
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from magcdw.hom.minimize import find_minimum  # noqa: E402

MU_MAX, B_MAX = 2.0, 7.0


def row(args):
    B, mus, T = args
    out = []
    for mu in mus:
        r = find_minimum(B, mu, T)
        out.append({"mass": r.mass, "omega_rel": r.omega_rel, "phase": r.phase,
                    "minima": sum(1 for _, k in r.extrema if k == "min")})
    return out


def compute(T, n, workers, mu_range=(0.0, MU_MAX), b_range=(0.0, B_MAX)):
    """n x n grid; the lower ends are excluded (B > 0 is required)."""
    mus = mu_range[0] + (mu_range[1] - mu_range[0]) * np.arange(1, n + 1) / n
    fields = b_range[0] + (b_range[1] - b_range[0]) * np.arange(1, n + 1) / n
    start = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(row, [(B, mus, T) for B in fields], chunksize=4))
    return {"T": T, "g": "-pi", "mu": mus.tolist(), "B": fields.tolist(),
            "elapsed_s": time.perf_counter() - start, "rows": rows}


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    mus, fields = np.array(data["mu"]), np.array(data["B"])
    phases = sorted({c["phase"] for r in data["rows"] for c in r},
                    key=lambda p: (p != "V", p == "T", int(p[1:]) if p.startswith("I") else 0))
    code = {p: k for k, p in enumerate(phases)}
    # rows are fields -> transpose to [mu, B]
    P = np.array([[code[c["phase"]] for c in r] for r in data["rows"]]).T
    Mstar = np.array([[c["mass"] for c in r] for r in data["rows"]]).T
    dm, db = mus[1] - mus[0], fields[1] - fields[0]
    extent = (fields[0] - db, fields[-1], mus[0] - dm, mus[-1])
    colors = []
    for p in phases:
        if p == "V":
            colors.append("#2f5d9a")
        elif p == "T":
            colors.append("#e8e4d8")
        else:
            j = int(p[1:])
            colors.append(plt.cm.YlGn(0.35 + 0.6 * min(j, 8) / 8))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.imshow(P, origin="lower", aspect="auto", interpolation="nearest", extent=extent,
              cmap=ListedColormap(colors), vmin=-0.5, vmax=len(phases) - 0.5)
    for p, c in zip(phases, colors):
        ax.scatter([], [], marker="s", color=c, label=p)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              ncol=1 if len(phases) <= 12 else 2, frameon=False, handlelength=1)
    ax.set_title(f"Homogeneous phases, T = {data['T']}")
    ax = axes[1]
    im = ax.imshow(Mstar, origin="lower", aspect="auto", interpolation="nearest", extent=extent,
                   cmap="viridis")
    fig.colorbar(im, ax=ax, label=r"$M^*/m_*$")
    ax.set_title(r"$M^*$ of the homogeneous minimum")
    for ax in axes:
        ax.set_xlabel(r"$|eB|/m_*^2$")
        ax.set_ylabel(r"$\mu/m_*$")
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--mu", type=float, nargs=2, default=(0.0, MU_MAX))
    parser.add_argument("--B", type=float, nargs=2, default=(0.0, B_MAX))
    args = parser.parse_args()
    tag = "T0" if args.T == 0 else f"T{args.T:g}"
    if tuple(args.mu) != (0.0, MU_MAX) or tuple(args.B) != (0.0, B_MAX):
        tag += f"_mu{args.mu[0]:g}-{args.mu[1]:g}_B{args.B[0]:g}-{args.B[1]:g}"
    out = HERE / f"hom-diagram-{tag}.json"
    if not args.plot_only:
        data = compute(args.T, args.n, args.workers, tuple(args.mu), tuple(args.B))
        out.write_text(json.dumps(data))
        print(f"T={args.T}: {args.n}x{args.n} points in {data['elapsed_s']:.0f}s")
    plot(json.loads(out.read_text()), HERE.parent / "figures" / f"hom-diagram-{tag}.png")


if __name__ == "__main__":
    main()
