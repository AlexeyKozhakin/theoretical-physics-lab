"""Fast, explicitly provisional mu/B map at fixed T, with checked CDW points.

A shared spectral bank serves every mu on one B row. Local searches guide
candidate selection; only full basis bounds may label a point confirmed CDW.
Other labels describe approximate leaders of a finite search, not exclusion
of competing minima. Field rows may run in separate CPU processes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from adaptive_search import json_finite
from basis_bounds import finite_first_vacuum, infinite_first_vacuum, relative_interval
from homogeneous_potential import ladder_thermal_sum, minimize_mass, potential, thermal_weight, vacuum
from refine_cdw_candidate import CorrectedObjective
from relative_potential import RelativeBlock


FIELDS = [0.05, 0.075, 0.1, 0.125, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24,
          0.26, 0.28, 0.3, 0.325, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
CHEMICALS = np.round(np.linspace(0.8, 1.4, 25), 8).tolist()


def local_search(objective, seed):
    result = minimize(objective, seed, method="Nelder-Mead", bounds=((0, 1.6), (0, 1.7)),
                      options={"xatol": 3e-4, "fatol": 1e-9, "maxiter": 100})
    M, b = (float(x) for x in result.x)
    return {"mass": M, "wave": b, "corrected_gap": float(result.fun),
            "converged": bool(result.success), "evaluations": int(result.nfev),
            "outer_boundary": M >= 1.599 or b >= 1.699}


class HomogeneousRow:
    """Reuse the vacuum values on a common mass grid for all mu in a row."""
    def __init__(self, field, temperature):
        self.B, self.T = field, temperature
        self.masses = np.linspace(0, 4, 81)
        self.vacuum = np.array([vacuum(field, M) for M in self.masses])
        nmax = math.ceil((max(CHEMICALS)+40*temperature)**2/(2*field))
        n = np.arange(nmax+1)
        self.energies = np.sqrt(self.masses[:, None]**2+2*field*n)
        self.weights = np.r_[1, np.full(nmax, 2)]

    def minimum(self, mu):
        values = self.vacuum-self.B/(2*math.pi)*np.sum(
            thermal_weight(self.energies, mu, self.T)*self.weights, axis=1)
        points = [(0.0, float(values[0])), (4.0, float(values[-1]))]
        brackets = [(0, self.masses[1])]
        brackets += [(self.masses[i-1], self.masses[i+1]) for i in range(1, 80)
                     if values[i] <= values[i-1] and values[i] <= values[i+1]]
        for lower, upper in brackets:
            optimum = minimize_scalar(lambda M: potential(self.B, M, mu, self.T)[0],
                                      bounds=(lower, upper), method="bounded",
                                      options={"xatol": 2e-9})
            points.append((float(optimum.x), float(optimum.fun)))
        # Filling thresholds supplement the regular grid and can seed narrow
        # wells even when adjacent regular nodes both look unfavorable.
        for n in range(math.floor(mu**2/(2*self.B))+1):
            mass = math.sqrt(max(mu**2-2*self.B*n, 0))
            if 0 < mass < 4:
                optimum = minimize_scalar(lambda M: potential(self.B, M, mu, self.T)[0],
                                          bounds=(max(0, mass-.05), min(4, mass+.05)), method="bounded",
                                          options={"xatol": 2e-9})
                points.append((float(optimum.x), float(optimum.fun)))
        M, value = min(points, key=lambda p:p[1])
        if M < 1e-6:
            M, value = 0.0, potential(self.B, 0, mu, self.T)[0]
        return {"mass": M, "potential": value, "mass_max": 4,
                "method": "shared 81-node grid, threshold starts and local refinements"}


def spectral_bank(B, T, cutoff=64):
    masses = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.1, 1.4]
    waves = [0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
             0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    coordinates, energies, bases = [], [], []
    failures = []
    for M in masses:
        hom_vacuum = vacuum(B, M)
        for b in waves:
            try:
                block = RelativeBlock(B, M, b, cutoff)
                E = np.sqrt(block.eigenvalues())
                reference_E = np.sqrt(np.r_[block.upper, block.lower])
                vacuum_diff = -block.degeneracy*float(np.sum(E)-np.sum(reference_E))
                first_tail = infinite_first_vacuum(B, M, b).value-finite_first_vacuum(block)
                coordinates.append((M,b))
                energies.append(E)
                bases.append(hom_vacuum+vacuum_diff+first_tail)
            except (ArithmeticError, FloatingPointError) as error:
                failures.append({"mass": M, "wave": b, "reason": str(error)})
    if not coordinates:
        raise ArithmeticError("No usable spectral bank")
    # The omitted homogeneous thermal tail beyond N=64 is negligible on the
    # present B/mu/T window; full checks below do not use this approximation.
    return np.array(coordinates), np.array(energies), np.array(bases), failures


def check_candidate(B, mu, T, point, reference):
    rows = []
    for N in (128, 256, 512):
        block = RelativeBlock(B, point[0], point[1], N)
        interval = relative_interval(block, mu, T, frequency_tolerance=1e-9)
        hom, hom_error = potential(B, point[0], mu, T)
        direct = block.spectral_difference(mu,T)
        discrepancy = abs(direct-interval.finite_basis_value)
        if discrepancy > interval.frequency_error_bound+5e-10:
            raise ArithmeticError("Finite spectral/frequency results disagree")
        lower = interval.lower+hom-reference-hom_error-2e-9
        upper = interval.upper+hom-reference+2e-9
        rows.append({"N": N, "lower": lower, "upper": upper,
                     "direct_frequency_discrepancy": discrepancy,
                     "frequency_error_bound": interval.frequency_error_bound,
                     "vacuum_upper_method": interval.vacuum.upper_method})
        if N >= 256 and upper < -1e-8:
            break
    return rows


def run_field(B):
    start = time.monotonic()
    T = 0.02
    coordinates, energies, bases, bank_failures = spectral_bank(B,T)
    hom_row = HomogeneousRow(B,T)
    points = []
    previous = None
    for mu in CHEMICALS:
        reference = hom_row.minimum(mu)
        values = bases-B/(2*math.pi)*np.sum(thermal_weight(energies,mu,T),axis=1)-reference["potential"]
        order = np.argsort(values)
        seeds = [tuple(coordinates[order[0]])]
        # Explore a distinct bank well as well as the nearest chemical-potential
        # solution. Several seeds help avoid following only one local branch.
        for index in order[1:]:
            seed = tuple(coordinates[index])
            if abs(seed[0]-seeds[0][0])>.12 or abs(seed[1]-seeds[0][1])>.15:
                seeds.append(seed)
                break
        if previous is not None:
            seeds.append(previous)
        if reference["mass"]>1e-4:
            seeds.append((reference["mass"], .05))
        objective = CorrectedObjective(B,mu,T,64,reference["potential"])
        solutions = []
        failure = None
        validation = []
        refined_reference = None
        best = None
        try:
            for seed in seeds:
                solutions.append(local_search(objective,seed))
            best = min(solutions,key=lambda x:x["corrected_gap"])
            if best["corrected_gap"] < -2e-7 and best["mass"]>1e-4 and best["wave"]>1e-4:
                # A denser independent homogeneous search is mandatory before
                # any confirmed-CDW label. The shift does not move local minima.
                dense = minimize_mass(B,mu,T,grid_points=141,mass_max=6)
                refined_reference = asdict(dense)
                if dense.upper_boundary_selected:
                    raise ArithmeticError("Homogeneous reference hits upper mass boundary")
                if abs(dense.potential-reference["potential"])>1e-8:
                    dense2 = minimize_mass(B,mu,T,grid_points=401,mass_max=8)
                    if abs(dense2.potential-dense.potential)>1e-9:
                        raise ArithmeticError("Refined homogeneous searches disagree")
                    refined_reference = asdict(dense2)
                objective128 = CorrectedObjective(B,mu,T,128,refined_reference["potential"])
                best128 = local_search(objective128,(best["mass"],best["wave"]))
                best128["N"] = 128
                solutions.append(best128)
                best = best128
                if best["corrected_gap"] < -1e-8:
                    validation = check_candidate(B,mu,T,(best["mass"],best["wave"]),
                                                 refined_reference["potential"])
            if validation and validation[-1]["upper"] < -1e-8:
                status = "confirmed_cdw"
                previous = best["mass"],best["wave"]
            elif (best["corrected_gap"] < -1e-7 or best["outer_boundary"]
                  or not best["converged"] or bank_failures):
                status = "unresolved"
                previous = None
            else:
                status = ("homogeneous_broken_provisional" if reference["mass"]>1e-4
                          else "symmetric_provisional")
                previous = None
        except (ArithmeticError, FloatingPointError, ValueError) as error:
            status, failure = "unresolved", str(error)
            previous = None
            if best is None and solutions:
                best = min(solutions,key=lambda x:x["corrected_gap"])
        points.append({"B":B,"mu":mu,"T":T,"status":status,
                       "homogeneous_reference":reference,"refined_reference":refined_reference,
                       "bank_best_gap":float(values[order[0]]),"starts":[list(s) for s in seeds],
                       "local_solutions":solutions,"selected_point":best,"validation":validation,
                       "numerical_failure":failure})
    return {"B":B,"points":points,"bank_backgrounds":len(coordinates),"bank_failures":bank_failures,
            "elapsed_seconds":time.monotonic()-start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="cdw-magnetic/code/phase-scout.json")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    report = {"date":"2026-11-07","T":.02,"g":"-pi","completed":False,
              "field_axis":FIELDS,"chemical_axis":CHEMICALS,"rows":[],
              "workers":args.workers,"coarse_cutoff":64,"search_gap_tolerance":1e-7,
              "bank_homogeneous_thermal_tail_upper_max":max(
                  B/(2*math.pi)*ladder_thermal_sum(B,0,max(CHEMICALS),.02,start_rank=128).upper
                  for B in FIELDS),
              "search_domain":{"mass":[0,1.6],"wave":[0,1.7]},
              "interpretation":"Provisional finite-search map. Confirmed CDW means a checked fixed background "
                               "beats the computed homogeneous reference. Other phase labels are approximate "
                               "search leaders, never a global exclusion of CDW. No interpolated boundary is proved."}
    if args.resume and destination.exists():
        report = json.loads(destination.read_text())
        if report["field_axis"]!=FIELDS or report["chemical_axis"]!=CHEMICALS:
            raise ValueError("Resume grid differs from current grid")
        report["completed"] = False
    done = {row["B"] for row in report["rows"]}
    pending = [B for B in FIELDS if B not in done]
    destination.write_text(json.dumps(json_finite(report),indent=2,allow_nan=False)+"\n")
    # Processes perform independent numerical field rows; only this parent
    # writes the checkpoint. This does not use collaborating AI agents.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_field,B):B for B in pending}
        for job in as_completed(jobs):
            row = job.result()
            report["rows"].append(row)
            report["rows"].sort(key=lambda x:x["B"])
            destination.write_text(json.dumps(json_finite(report),indent=2,allow_nan=False)+"\n")
            counts = {s:sum(x["status"]==s for x in row["points"]) for s in sorted(set(x["status"] for x in row["points"]))}
            print(f"B={row['B']}: {counts}; {len(report['rows'])}/{len(FIELDS)} rows; "
                  f"{row['elapsed_seconds']:.1f}s",flush=True)
    report["completed"] = True
    report["counts"] = {s:sum(x["status"]==s for row in report["rows"] for x in row["points"])
                        for s in sorted(set(x["status"] for row in report["rows"] for x in row["points"]))}
    destination.write_text(json.dumps(json_finite(report),indent=2,allow_nan=False)+"\n")
    print(f"Wrote {destination}: {report['counts']}",flush=True)


if __name__=="__main__":
    main()
