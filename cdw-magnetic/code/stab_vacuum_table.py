"""Tabulate the scaled vacuum kernels of the second variation (note stability/05).

At eB = 1: x = q, y = M; Phi_pi_vac(x, y) and DeltaPi_vac(x, y) from magcdw.stab.landau.vacuum_kernels.
For any eB: Phi_pi_vac(q, M, B) = sqrt(B) Phi_pi_vac(q/sqrt(B), M/sqrt(B)), the same for DeltaPi.

Usage (from the repository root):
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/stab_vacuum_table.py --workers 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from magcdw.stab.landau import vacuum_kernels  # noqa: E402

X = np.r_[0.05, 0.1, 0.15, np.arange(0.2, 1.0001, 0.1), np.arange(1.2, 3.0001, 0.2), np.arange(3.5, 10.0001, 0.5)]
Y = np.r_[0.002, 0.05, np.arange(0.1, 1.0001, 0.1), np.arange(1.2, 3.0001, 0.2), np.arange(3.5, 6.0001, 0.5)]


def task(args):
    i, j = args
    return i, j, vacuum_kernels(float(X[i]), float(Y[j]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    phi = np.full((len(X), len(Y)), np.nan)
    delta = np.full((len(X), len(Y)), np.nan)
    start = time.perf_counter()
    jobs = [(i, j) for j in range(len(Y)) for i in range(len(X))]
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (i, j, (p, d)) in enumerate(pool.map(task, jobs, chunksize=4)):
            phi[i, j], delta[i, j] = p, d
            if k % 100 == 0:
                print(f"{k}/{len(jobs)} {time.perf_counter() - start:.0f}s", flush=True)
    out = ROOT / "magcdw" / "stab" / "data" / "vacuum_table.npz"
    np.savez(out, x=X, y=Y, phi_pi=phi, delta_pi=delta)
    print(f"done {len(jobs)} points in {time.perf_counter() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
