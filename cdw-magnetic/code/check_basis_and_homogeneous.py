"""Check analytic basis bounds and the independent homogeneous reference.

Run: OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_basis_and_homogeneous.py
Optional --output file.json saves diagnostics; it is not a phase diagram.
"""

import argparse
from dataclasses import asdict
import json
import math

import mpmath as mp
import numpy as np
from scipy.optimize import brentq

from basis_bounds import relative_interval, thermal_basis_bound, vacuum_basis_bound
from homogeneous_potential import ladder_thermal_sum, minimize_mass, potential, thermal_weight, vacuum
from relative_potential import RelativeBlock


def check_ladders():
    for offset in (0.49, -0.07):
        for rank in (0, 1, 2, 7):
            result = ladder_thermal_sum(0.8, offset, 1.8, 0.2, start_rank=rank)
            j = np.arange(rank, 2000)
            n = (j + 1) // 2
            energies = np.sqrt(np.maximum(offset + 1.6 * n, 0))
            reference = float(np.sum(thermal_weight(energies, 1.8, 0.2)))
            assert result.lower - 2e-13 <= reference <= result.upper + 2e-13
    print("PASS homogeneous/comparison ladders: ranks, partial degeneracies, clamped lower spectra, tails.")


def check_relative_bounds():
    results = []
    large = RelativeBlock(0.8, 0.7, 0.4, 256)
    large_spectrum = large.eigenvalues()
    for cutoff in (16, 32, 64):
        block = RelativeBlock(0.8, 0.7, 0.4, cutoff)
        for chemical, temperature in ((1.2, 0.2), (1.8, 0.2), (4.0, 0.15)):
            result = relative_interval(block, chemical, temperature)
            assert result.lower <= result.upper
            if chemical in (1.2, 1.8):
                assert result.fixed_background_sign == "positive_cost_at_fixed_mass"
            # Independently check the thermal spectral enclosure on a much
            # larger Ritz compression (no use of its uncontrolled vacuum sum).
            finite_trace = float(np.sum(thermal_weight(np.sqrt(block.eigenvalues()), chemical, temperature)))
            large_trace = float(np.sum(thermal_weight(np.sqrt(large_spectrum), chemical, temperature)))
            ref_missing = ladder_thermal_sum(block.field, block.mass ** 2, chemical, temperature,
                                             start_rank=2 * cutoff)
            thermal = result.thermal
            max_missing_trace = ref_missing.lower - thermal.lower_correction / block.degeneracy
            assert large_trace >= finite_trace - 2e-11
            assert large_trace <= finite_trace + max_missing_trace + 2e-11
            results.append({"N": cutoff, "mu": chemical, "T": temperature, **asdict(result)})
            print(f"bound N={cutoff}, mu={chemical}: raw={result.finite_basis_value:.10g}, "
                  f"infinite=[{result.lower:.10g},{result.upper:.10g}], "
                  f"{result.fixed_background_sign}")

    # Verify the vacuum bounds against a genuinely infinite-basis heat-series
    # interval, not against an uncorrected larger finite vacuum matrix.
    from check_relative_potential import heat_vacuum_partial, heat_vacuum_tail_bound
    heat = heat_vacuum_partial(0.8, 0.8, 0.1, 44, 7)
    heat_tail = heat_vacuum_tail_bound(0.8, 0.8, 0.1)
    for cutoff in (16, 64):
        block = RelativeBlock(0.8, 0.8, 0.1, cutoff)
        correction = vacuum_basis_bound(block)
        raw = block.spectral_difference(0, 0)
        # The independent analytical interval and the basis interval must
        # overlap; quadrature has been cross-checked in the earlier script.
        assert raw + correction.lower <= heat + heat_tail + 3e-11
        assert raw + correction.upper >= heat - 3e-11
    strong = vacuum_basis_bound(RelativeBlock(0.2, 0.1, 0.8, 64))
    assert strong.supported and strong.kappa0 > 1
    assert strong.upper_method == "discarded_tail_schur"
    unsupported = vacuum_basis_bound(RelativeBlock(0.2, 0.7, 1.2, 2))
    assert not unsupported.supported and math.isinf(unsupported.upper)
    assert unsupported.lower >= 0
    zero = relative_interval(RelativeBlock(0.8, 0, 2, 12), 1.8, 0.2)
    assert zero.lower == zero.upper == 0
    tiny = relative_interval(RelativeBlock(0.8, 0.7, 1e-6, 16), 1.2, 0.2)
    assert tiny.fixed_background_sign == "unresolved"
    print("PASS basis bounds: independent vacuum heat interval, one-sided unsupported regime, exact zero.")
    return results


def check_homogeneous():
    rows = []
    assert abs(vacuum(0, 1) + 1 / (6 * math.pi)) < 1e-14
    below = minimize_mass(0, 0.8, 0, grid_points=61)
    above = minimize_mass(0, 1.2, 0, grid_points=61)
    assert abs(below.mass - 1) < 1e-6
    assert above.mass < 1e-6
    flat = [potential(0, M, 1, 0)[0] for M in np.linspace(0, 1, 21)]
    assert np.ptp(flat) < 1e-14
    print("PASS B=T=0: broken/symmetric minima and flat coexistence interval at mu=1.")

    critical_temperature = 1 / (2 * math.log(2))
    root = brentq(lambda M: M + 2 * 0.5 * math.log1p(math.exp(-M / 0.5)) - 1,
                  1e-8, 2)
    thermal_broken = minimize_mass(0, 0, 0.5, grid_points=61)
    thermal_restored = minimize_mass(0, 0, 0.8, grid_points=61)
    assert abs(thermal_broken.mass - root) < 1e-6
    assert thermal_restored.mass < 1e-5
    print(f"PASS B=mu=0: independent gap equation, T_c={critical_temperature:.12g}.")

    # Vacuum B->0 reference; this is independent of the general heat-density test.
    reference = vacuum(0, 0.7)
    errors = [abs(vacuum(B, 0.7) - reference) for B in (0.1, 0.05, 0.025)]
    assert all(first / second > 3.8 for first, second in zip(errors, errors[1:]))

    # Independent B=0 thermal expression in terms of polylogarithms.
    for M, mu, T in ((0.0, 1.2, 0.2), (0.7, 1.8, 0.1), (1.2, 0.8, 0.2)):
        with mp.workdps(40):
            m, u, t = (mp.mpf(str(x)) for x in (M, mu, T))
            exact_thermal = (m * t ** 2 / mp.pi * sum(mp.polylog(2, -mp.exp((sign * u - m) / t))
                                                     for sign in (1, -1))
                             + t ** 3 / mp.pi * sum(mp.polylog(3, -mp.exp((sign * u - m) / t))
                                                   for sign in (1, -1)))
        computed, error = potential(0, M, mu, T)
        assert abs(computed - vacuum(0, M) - float(exact_thermal.real)) < error + 2e-11
    print("PASS homogeneous potential: B->0 and independent thermal polylogarithms.")

    # Reference points only; not a grid of CDW/global two-parameter minima.
    for field, chemical, temperature in ((0.0, 0.8, 0.02), (0.0, 1.2, 0.02),
                                         (0.2, 0.0, 0.02), (0.8, 0.0, 0.02),
                                         (2.0, 0.0, 0.02), (0.8, 1.2, 0.02)):
        coarse = minimize_mass(field, chemical, temperature, grid_points=61)
        refined = minimize_mass(field, chemical, temperature, grid_points=121,
                                 mass_max=coarse.mass_interval[1] * 1.5)
        assert not coarse.upper_boundary_selected and not refined.upper_boundary_selected
        assert abs(coarse.potential - refined.potential) < 2e-9
        rows.append({"B": field, "mu": chemical, "T": temperature,
                     "coarse": asdict(coarse), "refined": asdict(refined)})
        print(f"homogeneous B={field:g}, mu={chemical:g}, T={temperature:g}: "
              f"M={refined.mass:.10g}, Omega={refined.potential:.10g}")
    return rows


def check_reference_modulations(minima):
    reference = minima[-1]
    assert reference["B"] == 0.8 and reference["mu"] == 1.2
    mass = reference["refined"]["mass"]
    rows = []
    for wave in (0.05, 0.1, 0.2, 0.4, 0.6):
        block = RelativeBlock(0.8, mass, wave, 24)
        result = relative_interval(block, 1.2, 0.02)
        assert result.lower > 0
        rows.append({"B": 0.8, "mu": 1.2, "T": 0.02, "M": mass, "b": wave,
                     **asdict(result)})
        print(f"reference modulation b={wave:g}: raw={result.finite_basis_value:.10g}, "
              f"infinite=[{result.lower:.10g},{result.upper:.10g}]")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    check_ladders()
    intervals = check_relative_bounds()
    minima = check_homogeneous()
    modulations = check_reference_modulations(minima)
    if args.output:
        from pathlib import Path
        report = {"normalization": "g=-pi, m_star=1; B means |eB|",
                  "relative_intervals": intervals, "homogeneous_reference_points": minima,
                  "modulations_at_reference_minimum": modulations,
                  "limitations": "Analytic inequalities plus estimated numerical quadrature; "
                                 "not interval-arithmetic certification; fixed backgrounds are not global phases."}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")
