"""Two continuation cuts through a CDW candidate, not a global phase map."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from adaptive_search import json_finite
from homogeneous_potential import minimize_mass, potential
from refine_cdw_candidate import CorrectedObjective, optimize_local, validate_point
from zero_field_potential import potential as zero_field_potential


def run_case(field, chemical, seed, temperature=0.02):
    coarse = minimize_mass(field, chemical, temperature, grid_points=81, mass_max=4)
    fine = minimize_mass(field, chemical, temperature, grid_points=141, mass_max=6)
    discrepancy = abs(coarse.potential - fine.potential)
    if discrepancy > 1e-9 or fine.upper_boundary_selected or coarse.upper_boundary_selected:
        raise ArithmeticError("Homogeneous searches disagree or touch boundary; refine reference")
    reference = fine.potential
    search_rows = []
    mass_bounds, wave_bounds = (0, 0.7), (0.2, 1.5)
    for N in (64, 128):
        objective = CorrectedObjective(field, chemical, temperature, N, reference)
        seeds = [seed, (0.12, 0.8)] if N == 64 else [seed]
        solutions = [optimize_local(objective, x, mass_bounds=mass_bounds, wave_bounds=wave_bounds)
                     for x in seeds]
        best = min(solutions, key=lambda x: x["corrected_gap"])
        if N == 64 and best["corrected_gap"] >= -1e-8:
            # A collapsed trajectory can miss another local well. Extra
            # starts diagnose that failure without claiming global exclusion.
            solutions += [optimize_local(objective, x, mass_bounds=mass_bounds, wave_bounds=wave_bounds)
                          for x in ((0.3, 0.65), (0.3, 0.9), (0.55, 0.6))]
            best = min(solutions, key=lambda x: x["corrected_gap"])
            if best["mass"] >= 0.6999:
                mass_bounds, wave_bounds = (0, 1.4), (0, 1.5)
                solutions += [optimize_local(objective, x, mass_bounds=mass_bounds, wave_bounds=wave_bounds)
                              for x in ((best["mass"], best["wave"]), (fine.mass, 0.05))]
                best = min(solutions, key=lambda x: x["corrected_gap"])
        seed = (best["mass"], best["wave"])
        search_rows.append({"N": N, "solutions": solutions})
    boundary = (best["mass"] > mass_bounds[1]-1e-4 or best["wave"] > wave_bounds[1]-1e-4
                or (wave_bounds[0] > 0 and best["wave"] < wave_bounds[0]+1e-4))
    # A collapsed branch and a missed branch must not be called a homogeneous
    # phase. Upper bounds in a fixed point can only establish a candidate.
    validation = []
    if best["mass"] > 1e-4 and best["corrected_gap"] < -1e-8:
        validation = validate_point(field, chemical, temperature, seed, reference, cutoffs=(128, 256))
        if validation[-1]["gap_upper"] >= -1e-8:
            validation += validate_point(field, chemical, temperature, seed, reference, cutoffs=(512,))
        status = ("confirmed_improving_point" if validation[-1]["gap_upper"] < -1e-8
                  else "unresolved_improving_search_value")
    elif best["mass"] <= 1e-4:
        status = "tracked_branch_collapsed_to_zero_mass"
    else:
        status = "no_improving_point_on_tracked_local_branch"
    if boundary:
        status += "_search_boundary"
    result = {"B": field, "mu": chemical, "T": temperature,
              "homogeneous_reference": asdict(fine), "reference_coarse_difference": discrepancy,
              "searches": search_rows, "selected_point": best, "validation": validation,
              "status": status, "search_boundary": boundary,
              "actual_search_window": {"mass": mass_bounds, "wave": wave_bounds}}
    print(f"B={field:g}, mu={chemical:g}: M={seed[0]:.7g}, b={seed[1]:.7g}, "
          f"search gap={best['corrected_gap']:.8g}, {status}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cdw-magnetic/code/cdw-branch-continuation.json")
    parser.add_argument("--recheck-unconfirmed", action="store_true",
                        help="Keep confirmed cases; retry other cases with additional starting points")
    args = parser.parse_args()
    if args.recheck_unconfirmed:
        report = json.loads(Path(args.output).read_text())
        report["completed"] = False
        results = {}
        for name in ("mu_cut", "field_cut"):
            for index, case in enumerate(report[name]):
                key = case["B"], case["mu"]
                if key not in results:
                    results[key] = (case if case["status"] == "confirmed_improving_point" else
                                    run_case(*key, seed=(0.11199624, 0.79953681)))
                report[name][index] = results[key]
                Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False)+"\n")
        report["completed"] = True
        report["unconfirmed_cases_rechecked_with_extra_starts"] = True
        Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False)+"\n")
        print(f"Updated {args.output}", flush=True)
        return
    report = {"date": "2026-11-07", "T": 0.02, "g": "-pi", "completed": False,
              "search_window": {"mass": [0, 0.7], "wave": [0.2, 1.5]},
              "mu_cut": [], "field_cut": [],
              "limitations": "Continuation of a local branch only. Negative upper bounds establish "
                             "improving fixed points; other statuses never exclude competing CDW minima. "
                             "Analytic basis bounds use estimated numerical arithmetic, not an interval proof."}

    def save():
        Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n")

    # Walk outward from the known well in both directions; each direction
    # restarts from the original witness rather than from a collapsed endpoint.
    central_seed = (0.11199624009403308, 0.7995368100527627)
    for values in ([1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.12],
                   [1.04, 1.03, 1.02, 1.01, 1.0, 0.99, 0.98]):
        seed = central_seed
        for chemical in values:
            result = run_case(0.2, chemical, seed)
            report["mu_cut"].append(result)
            if result["selected_point"]["mass"] > 1e-4:
                seed = result["selected_point"]["mass"], result["selected_point"]["wave"]
            save()
    for values in ([0.22, 0.24, 0.26, 0.28], [0.18, 0.16, 0.14, 0.12]):
        seed = central_seed
        for field in values:
            result = run_case(field, 1.05, seed)
            report["field_cut"].append(result)
            if result["selected_point"]["mass"] > 1e-4:
                seed = result["selected_point"]["mass"], result["selected_point"]["wave"]
            save()
    center = next(x for x in report["mu_cut"] if x["mu"] == 1.05)
    report["field_cut"].append(center)
    reference = minimize_mass(0, 1.05, 0.02, grid_points=141, mass_max=4)
    tighter_reference = minimize_mass(0, 1.05, 0.02, grid_points=401, mass_max=6)
    if abs(reference.potential - tighter_reference.potential) > 1e-9:
        raise ArithmeticError("Zero-field homogeneous reference has not converged")
    controls = []
    for M, b in (central_seed, (0.1, 0.8), (0.05, 0.8), (0.2, 0.8)):
        value, error = zero_field_potential(M, b, 1.05, 0.02)
        controls.append({"mass": M, "wave": b, "gap": value - reference.potential,
                         "quadrature_error_estimate": error})
    report["zero_field_control"] = {"mu": 1.05, "T": 0.02,
                                    "homogeneous_reference": asdict(reference),
                                    "refined_reference": asdict(tighter_reference),
                                    "fixed_backgrounds": controls,
                                    "scope": "Same backgrounds at B=0, not a global CDW exclusion."}
    report["mu_cut"].sort(key=lambda x: x["mu"])
    report["field_cut"].sort(key=lambda x: x["B"])
    report["completed"] = True
    save()
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
