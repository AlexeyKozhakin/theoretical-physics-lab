"""Independent algorithm/strong-tail checks and a CDW witness check.

Run: OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_adaptive_search.py
Optional --output writes diagnostics, not a global phase diagram.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from adaptive_search import CandidateEvaluator, adaptive_mesh, classify, json_finite
from basis_bounds import infinite_first_vacuum, relative_interval, tail_schur_vacuum_upper
from homogeneous_potential import minimize_mass, potential
from relative_potential import RelativeBlock


def check_refinement():
    def toy(M, b):
        value = (M - 0.6) ** 2 + (b - 0.8) ** 2 - 0.1
        return {"mass": M, "wave": b, "lower_gap": value - 1e-9,
                "upper_gap": value + 1e-9, "status": classify(value - 1e-9, value + 1e-9, 1e-8)}
    initial = adaptive_mesh(toy, [0, 1, 2], [0, 1, 2], rounds=0)
    refined = adaptive_mesh(toy, [0, 1, 2], [0, 1, 2], rounds=1)
    assert not any(p["status"] == "improving_candidate" for p in initial)
    assert any(p["status"] == "improving_candidate" for p in refined)

    def uncertain(M, b):
        return {"mass": M, "wave": b, "lower_gap": -1, "upper_gap": 1,
                "raw_finite_gap": -0.5, "status": classify(-1, 1, 1e-8)}
    records = adaptive_mesh(uncertain, [0, 1], [0, 1], rounds=2)
    assert all(p["status"] == "unresolved" for p in records)
    print("PASS mesh: refinement finds an off-grid well; a negative raw value never resolves an interval.")


def check_tail_partition():
    # A finite larger operator independently checks each piece of the new
    # infinite upper bound: tail-internal cost and finite-rank boundary cost.
    small = RelativeBlock(0.2, 0.1, 0.8, 16)
    big = RelativeBlock(0.2, 0.1, 0.8, 100)
    n, m = small.cutoff, big.cutoff
    matrix = np.block([[np.diag(big.upper), big.v * big.D],
                       [big.v * big.D.T, np.diag(big.lower)]])
    tail_indices = np.r_[np.arange(n, m), np.arange(m + n, 2 * m)]
    tail_values = eigvalsh(matrix[np.ix_(tail_indices, tail_indices)])
    tail_reference = np.r_[big.upper[n:], big.lower[n:]]
    tail_cost = -big.degeneracy * float(np.sum(np.sqrt(tail_values)) - np.sum(np.sqrt(tail_reference)))
    boundary_cost = big.degeneracy * float(np.sum(np.sqrt(small.eigenvalues()))
                                          + np.sum(np.sqrt(tail_values))
                                          - np.sum(np.sqrt(big.eigenvalues())))
    bound = tail_schur_vacuum_upper(small)
    assert bound is not None
    assert -1e-11 <= tail_cost <= bound["tail"] + 1e-11
    assert -1e-11 <= boundary_cost <= bound["boundary"] + 1e-11
    total_difference = big.spectral_difference(0, 0) - small.spectral_difference(0, 0)
    assert abs(total_difference - tail_cost - boundary_cost) < 2e-11
    print("PASS discarded-tail Schur bound: independent finite tail and boundary partition.")
    return {"finite_tail_cost": tail_cost, "finite_boundary_cost": boundary_cost,
            "infinite_upper_bound": bound}


def check_small_mass():
    B, b = 0.8, 0.4
    C = B / (4 * math.pi) * (-math.expm1(-2 * b ** 2 / B))
    remainders = []
    for M in (1e-3, 1e-4, 1e-6):
        result = infinite_first_vacuum(B, M, b)
        remainder = (result.value - C * M) / M ** 2
        assert 0.0049 < remainder < 0.0051
        remainders.append(remainder)
    assert abs(remainders[-1] - remainders[-2]) < 1e-8
    print("PASS small mass: analytic LLL term plus nonzero O(M^2) correction, no hidden quadrature scale.")


def check_witness():
    B, mu, T, M, b = 0.2, 1.05, 0.02, 0.1, 0.8
    references = []
    for points, maximum in ((141, 4.0), (401, 6.0), (801, 8.0)):
        minimum = minimize_mass(B, mu, T, grid_points=points, mass_max=maximum)
        assert minimum.mass < 1e-6
        references.append({"grid_points": points, "mass_max": maximum, **asdict(minimum)})
    reference_value = potential(B, 0, mu, T)[0]
    assert max(abs(row["potential"] - reference_value) for row in references) < 2e-10
    hom_gap = potential(B, M, mu, T)[0] - reference_value
    intervals = []
    for cutoff in (64, 128, 256):
        block = RelativeBlock(B, M, b, cutoff)
        interval = relative_interval(block, mu, T, frequency_tolerance=1e-10)
        direct = block.spectral_difference(mu, T)
        assert abs(direct - interval.finite_basis_value) < interval.frequency_error_bound + 3e-10
        lower, upper = interval.lower + hom_gap - 2e-9, interval.upper + hom_gap + 2e-9
        if cutoff >= 128:
            assert upper < -5e-6
        intervals.append({"N": cutoff, "gap_lower": lower, "gap_upper": upper,
                          "finite_spectral_gap": direct + hom_gap, **asdict(interval)})
        print(f"witness N={cutoff}: gap=[{lower:.12g},{upper:.12g}], direct/Matsubara checked")
    evaluator = CandidateEvaluator(0.8, 1.2, 0.02, cutoffs=(16, 32, 64))
    false_signal = evaluator.evaluate(evaluator.reference["mass"], 0.1)
    assert false_signal["raw_finite_gap"] < 0
    assert false_signal["status"] == "sample_above_homogeneous_reference"
    print("PASS witness: negative upper bound survives N=128,256 and denser homogeneous searches; known false signal excluded.")
    return {"B": B, "mu": mu, "T": T, "M": M, "b": b,
            "homogeneous_references": references, "homogeneous_mass_gap": hom_gap,
            "intervals": intervals, "known_false_signal": false_signal,
            "interpretation": "Numerically controlled candidate below all computed homogeneous minima; "
                              "not a continuum interval proof or a complete phase boundary."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="cdw-magnetic/code/adaptive-search-checks.json")
    args = parser.parse_args()
    check_refinement()
    partition = check_tail_partition()
    check_small_mass()
    witness = check_witness()
    report = {"tail_partition": partition, "witness": witness,
              "numerical_status": "Analytic tail inequalities plus estimated quadrature and floating arithmetic."}
    Path(args.output).write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")
