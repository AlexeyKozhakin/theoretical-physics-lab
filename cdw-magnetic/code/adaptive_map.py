"""Adaptive (mu, B) phase map at fixed T with the fast engine (note 18).

Coarse 60 x 60 grid, then edge bisection: whenever two neighbouring nodes
(along mu or along B) carry different labels, the midpoint is computed.
Three bisection levels refine the step near boundaries by a factor of 8.
Labels: cdw (confirmed directly without the spline), hom (homogeneous, M > 0),
sym (M = 0). Values are not certified (see fast_engine.py).

Usage (from the repository root):
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/adaptive_map.py --workers 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np

from fast_engine import scan_row

HERE = Path(__file__).resolve().parent
T = 0.01
MU_MAX, B_MAX = 2.0, 7.0
COARSE = 60
LEVELS = 3
UNIT = 2 ** LEVELS                       # lattice units per coarse step
MASS_MAX, WAVE_MAX, STEP = 2.0, 2.0, 0.05
ENERGY_MAX = MU_MAX + 40 * T
CACHE = Path("/tmp") / "cdw-fast-engine-cache"


def mu_of(i):
    return MU_MAX * i / (COARSE * UNIT)


def b_of(j):
    return B_MAX * j / (COARSE * UNIT)


def wave_cap(field):
    """b range: WAVE_MAX, limited at weak field so the vacuum basis stays <= 512."""
    return min(WAVE_MAX, math.sqrt(30 * field))


def label(entry):
    if entry["status"] == "cdw":
        return "cdw"
    return "hom" if entry["hom_mass"] > 1e-4 else "sym"


def row_task(args):
    j, indices = args
    field = b_of(j)
    cap = wave_cap(field)
    masses = np.linspace(0, MASS_MAX, int(round(MASS_MAX / STEP)) + 1)
    waves = np.linspace(0, cap, max(3, int(round(cap / STEP)) + 1))
    start = time.perf_counter()
    _, results = scan_row(field, [mu_of(i) for i in indices], T, masses=masses, waves=waves,
                          cache_dir=CACHE, energy_max=ENERGY_MAX)
    out = {}
    for i, r in zip(indices, results):
        r = {k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in r.items()}
        if "direct_gaps" in r:
            r["direct_gaps"] = [float(x) for x in r["direct_gaps"]]
        r["B"] = field
        r["label"] = label(r)
        r["wave_cap"] = cap
        r["at_wave_cap"] = bool(r["label"] == "cdw" and r["wave"] >= cap - 1e-3)
        out[(i, j)] = r
    return j, out, time.perf_counter() - start


def run_level(tasks, nodes, workers, log):
    by_row = {}
    for i, j in tasks:
        by_row.setdefault(j, []).append(i)
    jobs = [(j, sorted(set(ii))) for j, ii in sorted(by_row.items())]
    # Weak-field rows are the slowest: start them first.
    jobs.sort(key=lambda job: job[0])
    with ProcessPoolExecutor(workers) as pool:
        for j, out, dt in pool.map(row_task, jobs):
            nodes.update(out)
            log(f"  B={b_of(j):.4f}: {len(out)} nodes, {dt:.1f}s")
    return len(jobs)


def refinement(nodes, step):
    """Midpoints of edges of length `step` whose end labels differ."""
    half = step // 2
    new = set()
    for (i, j), r in nodes.items():
        for di, dj in ((step, 0), (0, step)):
            other = nodes.get((i + di, j + dj))
            if other is not None and other["label"] != r["label"]:
                mid = (i + di // 2, j + dj // 2)
                if mid not in nodes:
                    new.add(mid)
    return new, half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    nodes = {}
    coarse = [(i * UNIT, j * UNIT) for j in range(1, COARSE + 1) for i in range(1, COARSE + 1)]
    log(f"coarse grid {COARSE}x{COARSE}: mu in (0,{MU_MAX}], eB in (0,{B_MAX}], T={T}")
    rows = run_level(coarse, nodes, args.workers, log)
    stats = [{"level": 0, "new_nodes": len(coarse), "rows": rows,
              "elapsed_s": time.perf_counter() - t0}]
    step = UNIT
    for level in range(1, LEVELS + 1):
        new, half = refinement(nodes, step)
        log(f"level {level}: {len(new)} new nodes")
        rows = run_level(new, nodes, args.workers, log) if new else 0
        stats.append({"level": level, "new_nodes": len(new), "rows": rows,
                      "elapsed_s": time.perf_counter() - t0})
        step = half
    total = time.perf_counter() - t0
    counts = {}
    for r in nodes.values():
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    log(f"done: {len(nodes)} nodes in {total:.0f}s; labels {counts}")
    report = {"T": T, "g": "-pi", "mu_max": MU_MAX, "B_max": B_MAX, "coarse": COARSE,
              "levels": LEVELS, "mass_max": MASS_MAX, "wave_max": WAVE_MAX, "grid_step": STEP,
              "interpretation": "Fast-engine map, not certified. cdw = negative gap confirmed "
                                "without the spline at two basis sizes; hom/sym = homogeneous "
                                "leader of a global grid search over (M, b).",
              "elapsed_s": total, "stats": stats, "counts": counts,
              "nodes": [{"i": i, "j": j, **r} for (i, j), r in sorted(nodes.items())]}
    (HERE / "adaptive-map.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
