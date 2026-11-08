"""Adaptive stability map of the homogeneous ground state (note stability/06, section 5).

Coarse grid plus edge bisection: whenever two neighbouring nodes (along mu or B) carry
different labels, the midpoint is computed; three levels refine the step by 8.
Labels: S (stable), U0 (M0 = 0 unstable), UG (Goldstone unstable), UR (radial unstable).

Usage (from the repository root):
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/stab_adaptive_map.py --workers 4
    MPLCONFIGDIR=/tmp/mpl python3 cdw-magnetic/code/stab_adaptive_map.py --plot-only
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

T = 0.01
MU_MIN, MU_MAX, B_MIN, B_MAX = 1.0, 2.0, 0.16, 2.0   # B >= 0.16: q <= 4 stays inside the vacuum table (q <= 10 sqrt(eB))
COARSE, LEVELS = 60, 3
UNIT = 2 ** LEVELS
QS = np.linspace(0.05, 4.0, 80)
OUT = HERE / "stab-adaptive-map.json"


def mu_of(i):
    return MU_MIN + (MU_MAX - MU_MIN) * i / (COARSE * UNIT)


def b_of(j):
    return B_MIN + (B_MAX - B_MIN) * j / (COARSE * UNIT)


def node(args):
    i, j = args
    mu, B = mu_of(i), b_of(j)
    hm = find_minimum(B, mu, T)
    M0 = hm.mass
    out = {"i": i, "j": j, "mu": mu, "B": B, "M0": M0, "phase": hm.phase}
    if M0 > 1e-4:
        gold, rad = map(np.array, zip(*(fast.modes(q, M0, B, mu, T) for q in QS)))
        ig, ir = int(np.argmin(gold)), int(np.argmin(rad))
        out.update(gold_min=float(gold[ig]), q_gold=float(QS[ig]),
                   rad_min=float(rad[ir]), q_rad=float(QS[ir]))
        least = min(out["gold_min"], out["rad_min"])
        out["min"] = least
        out["q"] = out["q_gold"] if out["gold_min"] <= out["rad_min"] else out["q_rad"]
        out["label"] = "S" if least >= 0 else ("UG" if out["gold_min"] <= out["rad_min"] else "UR")
    else:
        sym = np.array([fast.symmetric_kernel(q, B, mu, T, regulator_mass=0.005 * math.sqrt(B)) for q in QS])
        k = int(np.argmin(sym))
        out.update(min=float(sym[k]), q=float(QS[k]), label="S" if sym[k] >= 0 else "U0")
    return out


def run(tasks, nodes, workers, log):
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(node, sorted(tasks), chunksize=8):
            nodes[(r["i"], r["j"])] = r
    log(f"  {len(tasks)} nodes computed")


def refinement(nodes, step):
    new = set()
    for (i, j), r in nodes.items():
        for di, dj in ((step, 0), (0, step)):
            other = nodes.get((i + di, j + dj))
            if other is not None and other["label"] != r["label"]:
                mid = (i + di // 2, j + dj // 2)
                if mid not in nodes:
                    new.add(mid)
    return new


def plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    from scipy.spatial import cKDTree

    nodes = data["nodes"]
    order = ["S", "U0", "UG", "UR"]
    colors = ["#dfe6ee", "#8e44ad", "#c0392b", "#e67e22"]
    names = ["stable", "unstable, M=0", "unstable Goldstone", "unstable radial"]
    ij = np.array([[n["i"], n["j"]] for n in nodes], float)
    unit = data["coarse"] * 2 ** data["levels"]
    gi, gj = np.meshgrid(np.arange(1, unit + 1), np.arange(1, unit + 1), indexing="ij")
    _, idx = cKDTree(ij).query(np.c_[gi.ravel(), gj.ravel()])
    lab = np.array([order.index(n["label"]) for n in nodes])[idx].reshape(gi.shape)
    val = np.array([n["min"] for n in nodes])[idx].reshape(gi.shape)
    extent = (data["B_min"], data["B_max"], data["mu_min"], data["mu_max"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    ax = axes[0]
    ax.imshow(lab, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
              cmap=ListedColormap(colors), vmin=-0.5, vmax=3.5)
    for c, name in zip(colors, names):
        ax.scatter([], [], marker="s", color=c, label=name)
    amap = HERE / "adaptive-map.json"
    if amap.exists():
        cdw = [n for n in json.loads(amap.read_text())["nodes"] if n["label"] == "cdw"
               and data["mu_min"] < n["mu"] <= data["mu_max"] and n["B"] <= data["B_max"]]
        ax.scatter([n["B"] for n in cdw], [n["mu"] for n in cdw], s=3, color="k",
                   label="CDW nodes, branch A")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Stability of the homogeneous minimum, T = {data['T']} ({len(nodes)} nodes)")
    ax = axes[1]
    scale = max(1e-6, float(np.percentile(np.abs([n["min"] for n in nodes]), 90)))
    im = ax.imshow(np.clip(val, -scale, scale), origin="lower", aspect="auto", extent=extent,
                   interpolation="nearest", cmap="RdBu", norm=TwoSlopeNorm(0, -scale, scale))
    fig.colorbar(im, ax=ax, label=r"$\min_q$ of the softest mode (clipped)")
    ax.set_title("Softest mode (red: unstable)")
    for ax in axes:
        ax.set_xlabel(r"$|eB|/m_*^2$")
        ax.set_ylabel(r"$\mu/m_*$")
    fig.savefig(path, dpi=150)

    # wave vector of the instability
    un = [n for n in nodes if n["label"] != "S"]
    if un:
        fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
        sc = ax.scatter([n["B"] for n in un], [n["mu"] for n in un], c=[n["q"] for n in un],
                        s=6, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=r"$q$ of the most unstable mode")
        ax.set_xlim(data["B_min"], data["B_max"])
        ax.set_ylim(data["mu_min"], data["mu_max"])
        ax.set_xlabel(r"$|eB|/m_*^2$")
        ax.set_ylabel(r"$\mu/m_*$")
        ax.set_title("Wave vector of the instability")
        fig.savefig(path.with_name(path.stem + "-q.png"), dpi=150)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    fig = HERE.parent / "figures" / "stab-adaptive-map.png"
    if args.plot_only:
        plot(json.loads(OUT.read_text()), fig)
        return
    t0 = time.perf_counter()
    nodes, stats = {}, []

    def log(msg):
        print(msg, flush=True)

    coarse = {(i * UNIT, j * UNIT) for i in range(1, COARSE + 1) for j in range(1, COARSE + 1)}
    log(f"coarse {COARSE}x{COARSE}: mu in ({MU_MIN},{MU_MAX}], eB in ({B_MIN},{B_MAX}], T={T}")
    run(coarse, nodes, args.workers, log)
    stats.append({"level": 0, "new": len(coarse), "elapsed_s": time.perf_counter() - t0})
    step = UNIT
    for level in range(1, LEVELS + 1):
        new = refinement(nodes, step)
        log(f"level {level}: {len(new)} new nodes")
        if new:
            run(new, nodes, args.workers, log)
        stats.append({"level": level, "new": len(new), "elapsed_s": time.perf_counter() - t0})
        step //= 2
    counts = {}
    for r in nodes.values():
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    data = {"T": T, "g": "-pi", "mu_min": MU_MIN, "mu_max": MU_MAX, "B_min": B_MIN, "B_max": B_MAX,
            "coarse": COARSE, "levels": LEVELS, "q_grid": [float(QS[0]), float(QS[-1]), len(QS)],
            "elapsed_s": time.perf_counter() - t0, "stats": stats, "counts": counts,
            "nodes": [nodes[k] for k in sorted(nodes)]}
    OUT.write_text(json.dumps(data))
    log(f"done: {len(nodes)} nodes in {data['elapsed_s']:.0f}s; {counts}")
    plot(data, fig)


if __name__ == "__main__":
    main()
