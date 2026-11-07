"""Refine the known CDW well using a Born-tail corrected search objective.

The search objective is an approximation, NOT the certified upper endpoint.
Independent full analytic basis bounds and Matsubara comparisons validate
the final fixed backgrounds. Local optimizer convergence is not globality.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from adaptive_search import json_finite
from basis_bounds import finite_first_vacuum, infinite_first_vacuum, relative_interval
from homogeneous_potential import minimize_mass, potential
from relative_potential import RelativeBlock


class CorrectedObjective:
    def __init__(self, field, chemical, temperature, cutoff, reference):
        self.field, self.chemical, self.temperature = field, chemical, temperature
        self.cutoff, self.reference = cutoff, reference
        self.cache = {}

    def __call__(self, point):
        mass, wave = (float(x) for x in point)
        key = (mass, wave)
        if key not in self.cache:
            if mass < 0 or wave < 0:
                raise ValueError("Search coordinates must be nonnegative")
            block = RelativeBlock(self.field, mass, wave, self.cutoff)
            homogeneous, _ = potential(self.field, mass, self.chemical, self.temperature)
            first = infinite_first_vacuum(self.field, mass, wave)
            correction = first.value - finite_first_vacuum(block)
            raw = homogeneous - self.reference + block.spectral_difference(
                self.chemical, self.temperature)
            self.cache[key] = {"mass": mass, "wave": wave, "raw_gap": raw,
                               "first_vacuum_tail": correction,
                               "corrected_gap": raw + correction,
                               "quadrature_error_estimate": first.quadrature_error_estimate}
        return self.cache[key]["corrected_gap"]


def optimize_local(objective, seed, mass_bounds=(0.01, 0.3), wave_bounds=(0.6, 1.0)):
    result = minimize(objective, seed, method="Nelder-Mead", bounds=(mass_bounds, wave_bounds),
                      options={"xatol": 2e-6, "fatol": 2e-12, "maxiter": 200})
    if not result.success:
        raise RuntimeError(f"Local optimizer failed: {result.message}")
    touches = any(min(abs(x - lower), abs(x - upper)) < 1e-4
                  for x, (lower, upper) in zip(result.x, (mass_bounds, wave_bounds)))
    return {"mass": float(result.x[0]), "wave": float(result.x[1]),
            "corrected_gap": float(result.fun), "success": bool(result.success),
            "evaluations": int(result.nfev), "iterations": int(result.nit),
            "touches_local_search_boundary": touches}


def validate_point(field, chemical, temperature, point, reference, cutoffs=(128, 256, 512)):
    mass, wave = point
    hom, hom_tail = potential(field, mass, chemical, temperature)
    rows = []
    for cutoff in cutoffs:
        start = time.monotonic()
        block = RelativeBlock(field, mass, wave, cutoff)
        interval = relative_interval(block, chemical, temperature, frequency_tolerance=1e-10)
        direct = block.spectral_difference(chemical, temperature)
        discrepancy = abs(interval.finite_basis_value - direct)
        if discrepancy > interval.frequency_error_bound + 5e-10:
            raise ArithmeticError("Finite Matsubara and direct spectral results disagree")
        first = infinite_first_vacuum(field, mass, wave)
        corrected = direct + hom - reference + first.value - finite_first_vacuum(block)
        reference_margin = 2e-9
        lower = interval.lower + hom - reference - hom_tail - reference_margin
        upper = interval.upper + hom - reference + reference_margin
        rows.append({"N": cutoff, "gap_lower": lower, "gap_upper": upper,
                     "corrected_gap": corrected, "direct_matsubara_discrepancy": discrepancy,
                     "interval": asdict(interval), "elapsed_seconds": time.monotonic() - start})
        print(f"validation N={cutoff}, M={mass:.8g}, b={wave:.8g}: "
              f"[{lower:.12g}, {upper:.12g}], corrected={corrected:.12g}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cdw-magnetic/code/cdw-candidate-refinement.json")
    args = parser.parse_args()
    B, mu, T = 0.2, 1.05, 0.02
    hom_references = [asdict(minimize_mass(B, mu, T, grid_points=nodes, mass_max=maximum))
                      for nodes, maximum in ((141, 4), (401, 6))]
    reference = potential(B, 0, mu, T)[0]
    if max(abs(item["potential"] - reference) for item in hom_references) > 2e-10:
        raise ArithmeticError("Homogeneous reference changed; inspect before using M=0")
    if any(item["mass"] > 1e-6 for item in hom_references):
        raise ArithmeticError("Homogeneous mass is nonzero; update reference")

    objectives = []
    optimizations = []
    seed = (0.1, 0.8)
    for N in (32, 64, 128, 256):
        objective = CorrectedObjective(B, mu, T, N, reference)
        seeds = ((0.05, 0.75), seed, (0.2, 0.9)) if N == 32 else (seed,)
        solutions = [optimize_local(objective, start) for start in seeds]
        best = min(solutions, key=lambda item: item["corrected_gap"])
        if any(item["touches_local_search_boundary"] for item in solutions):
            raise RuntimeError("Local search hits a boundary; enlarge before interpreting")
        optimizations.append({"N": N, "solutions": solutions,
                              "unique_objective_evaluations": len(objective.cache)})
        seed = best["mass"], best["wave"]
        objectives.append(objective)
        print(f"local N={N}: M={seed[0]:.10g}, b={seed[1]:.10g}, "
              f"corrected gap={best['corrected_gap']:.12g}", flush=True)

    # Curvature diagnostics of the converged search objective, not a proof
    # about the exact infinite-basis Hessian. Compare two finite step sizes.
    final = objectives[-1]
    curvature = []
    center = np.asarray(seed)
    for step in (1e-3, 5e-4):
        directions = np.eye(2) * step
        f0 = final(center)
        diagonal = [(final(center + d) - 2 * f0 + final(center - d)) / step ** 2
                    for d in directions]
        d, e = directions
        mixed = (final(center + d + e) - final(center + d - e)
                 - final(center - d + e) + final(center - d - e)) / (4 * step ** 2)
        hessian = np.array([[diagonal[0], mixed], [mixed, diagonal[1]]])
        curvature.append({"step": step, "hessian": hessian.tolist(),
                          "eigenvalues": np.linalg.eigvalsh(hessian).tolist()})

    # Convergence of coordinates and values is separate from the interval
    # sign check. A coarse upper endpoint may be looser while still negative.
    finest = min(optimizations[-1]["solutions"], key=lambda item: item["corrected_gap"])
    for item in optimizations[1:]:
        best = min(item["solutions"], key=lambda row: row["corrected_gap"])
        if max(abs(best[key] - finest[key]) for key in ("mass", "wave")) > 1e-4:
            raise ArithmeticError("Local optimizer coordinates have not converged")
        if abs(best["corrected_gap"] - finest["corrected_gap"]) > 3e-9:
            raise ArithmeticError("Local optimizer values have not converged")
    if not all(min(item["eigenvalues"]) > 0 for item in curvature):
        raise ArithmeticError("Corrected search objective is not locally convex")

    validation = validate_point(B, mu, T, seed, reference)
    if not all(item["gap_upper"] < -1e-8 for item in validation) or validation[-1]["gap_upper"] >= -1e-5:
        raise ArithmeticError("Refined candidate failed independent negative-upper-bound check")

    # A local rectangular mesh records the landscape without asserting that
    # unvisited points are excluded. It also checks the optimizer off its path.
    M_axis = np.linspace(0.02, 0.22, 25)
    b_axis = np.linspace(0.65, 0.95, 31)
    surface_objective = CorrectedObjective(B, mu, T, 64, reference)
    surface = [[surface_objective((M, b)) for M in M_axis] for b in b_axis]
    report = {"date": "2026-11-07", "B": B, "mu": mu, "T": T, "g": -math.pi,
              "normalization": "m_star=1; potential per unit area and per fermion copy",
              "homogeneous_references": hom_references, "homogeneous_reference_value": reference,
              "search_domain": {"mass": [0.01, 0.3], "wave": [0.6, 1.0]},
              "optimizations": optimizations, "best_local_point": {"mass": seed[0], "wave": seed[1]},
              "curvature_diagnostics": curvature, "validation": validation,
              "local_surface": {"N": 64, "mass_axis": M_axis.tolist(), "wave_axis": b_axis.tolist(),
                                "corrected_gap": surface},
              "status": "Local numerical minimum; independent analytic basis bounds confirm a CDW candidate. "
                        "Born-corrected values guide the search and are not upper bounds; "
                        "no continuous globality or interval-arithmetic certificate."}
    Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
