"""Adaptive finite-window candidate search; no global phase certificate.

Every point is compared with the SAME best homogeneous reference. Positive
point bounds do not exclude unsampled cells. Missing upper bounds remain
unresolved, and raw finite-matrix gains never define a CDW candidate.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from basis_bounds import relative_interval
from homogeneous_potential import minimize_mass, potential
from relative_potential import RelativeBlock


def classify(lower, upper, tolerance):
    if lower > tolerance:
        return "sample_above_homogeneous_reference"
    if upper < -tolerance:
        return "improving_candidate"
    return "unresolved"


class CandidateEvaluator:
    def __init__(self, field, chemical, temperature, coupling=-math.pi,
                 cutoffs=(16, 32, 64), tolerance=1e-8):
        self.field, self.chemical, self.temperature = field, chemical, temperature
        self.coupling = coupling
        self.cutoffs = tuple(cutoffs)
        self.tolerance = tolerance
        self.cache = {}
        coarse = minimize_mass(field, chemical, temperature, coupling, grid_points=81)
        reference = minimize_mass(field, chemical, temperature, coupling, grid_points=141,
                                  mass_max=coarse.mass_interval[1] * 1.3)
        if coarse.upper_boundary_selected or reference.upper_boundary_selected:
            raise RuntimeError("Homogeneous reference touches its mass boundary")
        reference_mass, reference_value = reference.mass, reference.potential
        zero_value, _ = potential(field, 0, chemical, temperature, coupling)
        zero_resolution = 1e-6
        if reference_mass < zero_resolution and abs(reference_value - zero_value) < tolerance:
            reference_mass, reference_value = 0.0, zero_value
        _, numerical_tail = potential(field, reference_mass, chemical, temperature, coupling)
        uncertainty = abs(reference.potential - coarse.potential) + numerical_tail + tolerance / 10
        self.reference = {"mass": reference_mass, "potential": reference_value,
                          "minimizer_reported_mass": reference.mass,
                          "zero_mass_resolution": zero_resolution,
                          "mass_interval": reference.mass_interval,
                          "coarse_reference_difference": abs(reference.potential - coarse.potential),
                          "numerical_uncertainty_estimate": uncertainty,
                          "status": "grid-refined homogeneous reference, not a continuum interval proof"}

    def evaluate(self, mass, wave):
        mass, wave = round(float(mass), 12), round(float(wave), 12)
        # M=0 has no physical modulation; avoid duplicate evaluations in b.
        key = (mass, 0.0 if mass == 0 else wave)
        if key in self.cache:
            return self.cache[key]
        hom, hom_tail = potential(self.field, mass, self.chemical, self.temperature, self.coupling)
        gap = hom - self.reference["potential"]
        reference_error = self.reference["numerical_uncertainty_estimate"]
        lower, upper = -math.inf, math.inf
        history = []
        status = "unresolved"
        raw_best = None
        if mass == 0 or wave == 0:
            lower, upper = gap - hom_tail - reference_error, gap + reference_error
            status = ("symmetric_representative" if mass == 0 else "homogeneous_reference_family")
        else:
            for cutoff in self.cutoffs:
                try:
                    interval = relative_interval(RelativeBlock(self.field, mass, wave, cutoff),
                                                 self.chemical, self.temperature,
                                                 frequency_tolerance=self.tolerance / 10,
                                                 decision_tolerance=self.tolerance)
                except ValueError as error:
                    if "Discarded block" not in str(error):
                        raise
                    history.append({"N": cutoff, "status": "increase_basis", "reason": str(error)})
                    continue
                except (ArithmeticError, FloatingPointError) as error:
                    history.append({"N": cutoff, "status": "numerical_failure", "reason": str(error)})
                    continue
                point_lower = gap - hom_tail - reference_error + interval.lower
                point_upper = gap + reference_error + interval.upper
                lower, upper = max(lower, point_lower), min(upper, point_upper)
                if lower > upper + self.tolerance:
                    raise ArithmeticError("Independent basis intervals do not overlap")
                raw_best = gap + interval.finite_basis_value
                history.append({"N": cutoff, "lower": point_lower, "upper": point_upper,
                                "raw_gap": raw_best,
                                "frequency_bound": interval.frequency_error_bound,
                                "vacuum_upper_available": interval.vacuum.supported,
                                "thermal_lower_correction": interval.thermal.lower_correction})
                status = classify(lower, upper, self.tolerance)
                if status == "sample_above_homogeneous_reference":
                    break
                # Any improvement is verified again with at least two bases.
                if status == "improving_candidate" and len(history) >= 2:
                    break
                if math.isfinite(upper) and upper - lower < self.tolerance:
                    break
            if not math.isfinite(lower):
                status = ("numerical_failure" if any(item.get("status") == "numerical_failure"
                                                     for item in history) else "insufficient_basis")
        result = {"mass": mass, "wave": 0.0 if mass == 0 else wave,
                  "lower_gap": lower, "upper_gap": upper,
                  "homogeneous_mass_gap": gap, "raw_finite_gap": raw_best,
                  "status": status, "basis_history": history,
                  "reference_uncertainty_estimate": reference_error}
        self.cache[key] = result
        return result


def adaptive_mesh(evaluate, mass_grid, wave_grid, rounds=2, centers_per_round=3):
    """Heuristic point refinement. No unvisited cell is certified excluded."""
    masses, waves = sorted(set(mass_grid)), sorted(set(wave_grid))
    records = {}

    def add(mass, wave):
        result = evaluate(mass, wave)
        records[(result["mass"], result["wave"])] = result

    for mass in masses:
        for wave in waves:
            add(mass, wave)
    dm = max(np.diff(masses)) if len(masses) > 1 else 0
    db = max(np.diff(waves)) if len(waves) > 1 else 0
    for _ in range(rounds):
        candidates = [point for point in records.values()
                      if point["mass"] > 0 and point["wave"] > 0]
        # Prioritize uncertain low bounds, but also refine the best excluded
        # samples when the initial mesh finds no improving point.
        ordered = sorted(candidates, key=lambda point: point["lower_gap"])
        centers = []
        for point in ordered:
            if not any(abs(point["mass"] - center["mass"]) < dm / 2
                       and abs(point["wave"] - center["wave"]) < db / 2 for center in centers):
                centers.append(point)
            if len(centers) == centers_per_round:
                break
        dm, db = dm / 2, db / 2
        mass_axis = sorted(set(point["mass"] for point in records.values()))
        wave_axis = sorted(set(point["wave"] for point in records.values()))

        def local_midpoints(value, axis):
            position = min(range(len(axis)), key=lambda index: abs(axis[index] - value))
            points = [value]
            if position > 0:
                points.append((value + axis[position - 1]) / 2)
            if position + 1 < len(axis):
                points.append((value + axis[position + 1]) / 2)
            return points

        for center in centers:
            for mass in local_midpoints(center["mass"], mass_axis):
                for wave in local_midpoints(center["wave"], wave_axis):
                    if masses[0] <= mass <= masses[-1] and waves[0] <= wave <= waves[-1]:
                        add(mass, wave)
    return list(records.values())


def run_case(field, chemical, temperature=0.02, mass_max=2.0, wave_max=2.0,
             grid_size=6, rounds=2, cutoffs=(16, 32, 64)):
    evaluator = CandidateEvaluator(field, chemical, temperature, cutoffs=cutoffs)
    masses = list(np.linspace(0, mass_max, grid_size)) + [0.05, 0.1, evaluator.reference["mass"]]
    masses = [value for value in masses if 0 <= value <= mass_max]
    # Filling thresholds seed sharp structures in the mass direction.
    for n in range(math.floor(chemical ** 2 / (2 * field)) + 1):
        M = math.sqrt(max(chemical ** 2 - 2 * field * n, 0))
        if M <= mass_max:
            masses.append(M)
    waves = list(np.linspace(0, wave_max, grid_size)) + [0.05, 0.1, 0.2]
    waves += [value for value in (chemical / 2, chemical) if value <= wave_max]
    waves = [value for value in waves if 0 <= value <= wave_max]
    points = adaptive_mesh(evaluator.evaluate, masses, waves, rounds)
    counts = {status: sum(point["status"] == status for point in points)
              for status in sorted(set(point["status"] for point in points))}
    improving = sorted((point for point in points if point["status"] == "improving_candidate"),
                       key=lambda point: point["upper_gap"])
    unresolved = sorted((point for point in points if point["status"] in
                         ("unresolved", "numerical_failure", "insufficient_basis")),
                        key=lambda point: point["lower_gap"])
    boundary_interest = [point for point in improving + unresolved
                         if abs(point["mass"] - mass_max) < 1e-10
                         or abs(point["wave"] - wave_max) < 1e-10]
    raw_negative_resolved = sum(point["raw_finite_gap"] is not None
                                and point["raw_finite_gap"] < -evaluator.tolerance
                                and point["status"] == "sample_above_homogeneous_reference"
                                for point in points)
    return {"B": field, "mu": chemical, "T": temperature,
            "domain": {"mass": [0, mass_max], "wave": [0, wave_max]},
            "homogeneous_reference": evaluator.reference, "point_counts": counts,
            "raw_negative_samples_excluded_after_bounds": raw_negative_resolved,
            "improving_candidates": improving, "lowest_unresolved_samples": unresolved[:8],
            "boundary_interest": boundary_interest, "points": points,
            "search_status": "improving_sample_found" if improving else "no_improving_sample_found",
            "limitations": "Finite adaptive point search, not a continuous global minimum or phase proof; "
                           "analytic basis inequalities use estimated quadrature; infinite upper bounds stay unresolved."}


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cdw-magnetic/code/adaptive-search-pilot.json")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-cutoff", type=int, default=64)
    args = parser.parse_args()
    cutoffs = tuple(n for n in (16, 32, 64, 128) if n <= args.max_cutoff)
    if not cutoffs:
        parser.error("--max-cutoff must be >=16")
    cases = ((0.2, 0.95), (0.2, 1.05), (0.8, 1.2), (0.8, 1.4), (2.0, 1.2), (2.0, 1.6))
    report = {"normalization": "g=-pi; m_star=1; B=|eB|", "temperature": 0.02,
              "completed": False,
              "cases": [], "status": "finite-window pilot, not a phase diagram"}
    for field, chemical in cases:
        result = run_case(field, chemical, rounds=args.rounds, cutoffs=cutoffs)
        report["cases"].append(result)
        Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n")
        print(f"B={field}, mu={chemical}: M_hom={result['homogeneous_reference']['mass']:.8g}, "
              f"{result['point_counts']}, false_raw_signals="
              f"{result['raw_negative_samples_excluded_after_bounds']}", flush=True)
    report["completed"] = True
    Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
