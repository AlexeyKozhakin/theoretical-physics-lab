"""Benchmark fast_engine against the saved phase_scout row; see fast_engine.py.

Usage (from the repository root):
    OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/fast_benchmark.py
"""

import json
from pathlib import Path
import time

import numpy as np

from fast_engine import scan_row

HERE = Path(__file__).resolve().parent
CACHE = Path("/tmp") / "cdw-fast-engine-cache"


def main():
    old = json.loads((HERE / "phase-scout.json").read_text())
    row_old = next(r for r in old["rows"] if abs(r["B"] - 0.2) < 1e-12)
    mus = [p["mu"] for p in row_old["points"]]
    report = {}
    for label, workers, cache in (("1 process, no cache", 1, None),
                                  ("4 processes, cold cache", 4, CACHE),
                                  ("warm cache", 4, CACHE)):
        if label == "4 processes, cold cache":
            for f in CACHE.glob("*.npy"):
                f.unlink()
        start = time.perf_counter()
        row, result = scan_row(0.2, mus, 0.02, workers=workers, cache_dir=cache)
        report[label] = {"total_s": time.perf_counter() - start,
                         **{k: v for k, v in row.timing.items()}}
    start = time.perf_counter()
    row, dense = scan_row(0.2, np.linspace(0.8, 1.4, 121), 0.02, workers=4, cache_dir=CACHE)
    report["121 mu, warm cache"] = {"total_s": time.perf_counter() - start, **row.timing}
    for T in (0.01, 0.005):
        start = time.perf_counter()
        row, _ = scan_row(0.2, mus, T, workers=4, cache_dir=CACHE)
        report[f"25 mu, T={T}, warm cache"] = {"total_s": time.perf_counter() - start, **row.timing}
    agreement = []
    for p, r in zip(row_old["points"], result):
        sols = [s for s in p.get("local_solutions", []) if s.get("mass", 0) > 1e-6]
        old_gap = min((s["corrected_gap"] for s in sols), default=None)
        agreement.append({"mu": r["mu"], "old_status": p["status"], "old_gap": old_gap,
                          "new_gap": r["gap"], "mass": r["mass"], "wave": r["wave"],
                          "hom_ref_diff": r["hom_potential"] - p["homogeneous_reference"]["potential"]})
    print(json.dumps({"old_row_seconds": row_old["elapsed_seconds"], "timing": report}, indent=1))
    for a in agreement:
        og = "—" if a["old_gap"] is None else f"{a['old_gap']:+.2e}"
        print(f"mu={a['mu']:.3f} {a['old_status']:<32} old {og:>10}  new {a['new_gap']:+.2e} "
              f"M={a['mass']:.4f} b={a['wave']:.4f} dref={a['hom_ref_diff']:+.0e}")
    cdw = [(round(r["mu"], 4), f"{r['gap']:.2e}") for r in dense if r["gap"] < -1e-7]
    print("121-mu row, gap < -1e-7:", len(cdw), cdw)


if __name__ == "__main__":
    main()
