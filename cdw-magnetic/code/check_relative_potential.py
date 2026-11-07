"""Verify the relative Schur determinant and expose finite-basis artifacts.

Run from repo root, preferably with BLAS threads capped for these small matrices:
OPENBLAS_NUM_THREADS=1 python3 cdw-magnetic/code/check_relative_potential.py

Optional: --output path.json writes the reproducible convergence report.
"""

import argparse
import json
import math

import mpmath as mp
import numpy as np
from scipy.linalg import expm
from scipy.special import roots_genlaguerre

from check_heat_kernel import gaussian_spin_trace_term
from relative_potential import RelativeBlock, displacement, fermi_resolvent


def verify_displacement():
    alpha = 0.7
    reference_cutoff, cutoff = 70, 12
    a = np.diag(np.sqrt(np.arange(1, reference_cutoff, dtype=float)), 1)
    reference = expm(alpha * (a.T - a))[:cutoff, :cutoff]
    projected = displacement(alpha, cutoff)
    error = float(np.max(np.abs(projected - reference)))
    assert error < 2e-13
    assert np.linalg.norm(projected, 2) <= 1 + 2e-13
    small_a = a[:cutoff, :cutoff]
    wrong = expm(alpha * (small_a.T - small_a))
    assert np.max(np.abs(projected - wrong)) > 0.1
    print(f"PASS displacement: projected matrix error={error:.3g}; "
          "exp of the small generator is demonstrably different.")


def verify_fermi_resolvent():
    mp.mp.dps = 55
    for chemical, temperature in ((0.0, 0.2), (0.8, 0.02), (1.8, 0.2)):
        values = np.array([0.0, 1e-12, 0.01, 0.8, 2.0])
        G, derivative = fermi_resolvent(values, chemical, temperature)
        mu, T = mp.mpf(str(chemical)), mp.mpf(str(temperature))

        def reference(lam):
            if lam == 0:
                return 1 / (4 * T * mp.cosh(mu / (2 * T)) ** 2)
            E = mp.sqrt(lam)
            return mp.sinh(E / T) / (2 * E * (mp.cosh(E / T) + mp.cosh(mu / T)))

        for index, value in enumerate(values):
            lam = mp.mpf(str(value))
            ref = reference(lam)
            if value == 0:
                C = 1 / mp.cosh(mu / (2 * T)) ** 2
                ref_derivative = C * (2 - 3 * C) / (48 * T ** 3)
            else:
                ref_derivative = mp.diff(reference, lam)
            assert abs(G[index] - float(ref)) < 1e-9
            assert abs(derivative[index] - float(ref_derivative)) < 1e-7
    print("PASS analytic frequency sum: resolvent and degenerate derivative, including lambda=0.")


def verify_finite_determinants():
    results = []
    for cutoff in (12, 24):
        block = RelativeBlock(0.8, 0.7, 0.4, cutoff)
        for chemical, temperature in ((0.0, 0.2), (1.2, 0.2), (1.8, 0.2),
                                      (4.0, 0.15), (0.8, 0.02)):
            computed = block.matsubara(chemical, temperature, tolerance=1e-9)
            reference = block.spectral_difference(chemical, temperature)
            error = abs(computed.value - reference)
            assert error < computed.frequency_tail_bound + 2e-11
            results.append({"cutoff": cutoff, "mu": chemical, "T": temperature,
                            "delta": computed.value, "spectral_reference": reference,
                            "frequency_bound": computed.frequency_tail_bound,
                            "positive_frequencies": computed.positive_frequencies,
                            "error": error})
        vacuum, integration_error = block.vacuum()
        reference = block.spectral_difference(0, 0)
        assert vacuum > 0
        assert abs(vacuum - reference) < integration_error + 2e-10
    for mass, wave in ((0.0, 0.7), (0.7, 0.0), (0.0, 1000.0)):
        block = RelativeBlock(0.8, mass, wave, 12)
        assert block.matsubara(1.8, 0.2).value == 0
        assert block.vacuum()[0] == 0
    print("PASS Schur/Matsubara: 10 finite-matrix comparisons, two vacuum integrals, exact vanishing cases.")
    return results


def heat_vacuum_partial(field, mass, wave, proper_nodes, spin_nodes):
    x, weights = roots_genlaguerre(proper_nodes, -0.5)
    values = []
    for point in x:
        proper = point / mass ** 2
        t = 2 * field * proper
        terms = math.fsum(gaussian_spin_trace_term(
            t, wave / math.sqrt(2 * field), wave * mass / field, 0.5, r,
            nodes=(1, 24, 10, spin_nodes)[r]) for r in range(1, 4))
        values.append(math.exp(-(wave ** 2 + field) * proper) * terms / point)
    return field * mass / (4 * math.pi ** 1.5) * float(weights @ values)


def heat_vacuum_tail_bound(field, mass, wave):
    # Integrated factorial bound exists here because v<M^2.
    mp.mp.dps = 55
    B, M, b = (mp.mpf(str(x)) for x in (field, mass, wave))
    v = 2 * b * M
    assert v < M ** 2
    nu = M ** 2 / (2 * B)
    terms = []
    for r in range(4, 65):
        p = 2 * r - mp.mpf("0.5")
        integral = mp.gamma(p) / (2 * B) ** p * (2 * mp.zeta(p, nu) - nu ** (-p))
        terms.append(v ** (2 * r) / mp.factorial(2 * r) * integral)
    return float(B / (4 * mp.pi ** mp.mpf("1.5")) * mp.fsum(terms))


def verify_basis_convergence():
    rows = []
    for cutoff in (16, 32, 64, 128, 256, 512):
        block = RelativeBlock(0.8, 0.7, 0.4, cutoff)
        vacuum = block.spectral_difference(0, 0)
        delta = block.spectral_difference(1.2, 0.2)
        leading_tail = block.v ** 2 / (8 * math.pi * math.sqrt(2 * block.field * cutoff))
        rows.append({"cutoff": cutoff, "vacuum_delta": vacuum, "delta_mu_1p2_T_0p2": delta,
                     "leading_uv_tail_estimate": leading_tail,
                     "corrected_delta_estimate": delta + leading_tail})
        print(f"basis N={cutoff}: vacuum={vacuum:.10g}, delta={delta:.10g}, "
              f"leading-tail-corrected={delta+leading_tail:.10g}")
    assert rows[0]["delta_mu_1p2_T_0p2"] < 0 < rows[1]["delta_mu_1p2_T_0p2"]
    assert all(b["vacuum_delta"] > a["vacuum_delta"] for a, b in zip(rows, rows[1:]))
    print("PASS artifact detection: increasing the basis reverses the apparent CDW preference.")

    # Independent infinite-basis heat-series benchmark in a controlled regime.
    field, mass, wave = 0.8, 0.8, 0.1
    coarse = heat_vacuum_partial(field, mass, wave, 28, 6)
    refined = heat_vacuum_partial(field, mass, wave, 44, 7)
    quadrature_change = abs(coarse - refined)
    bound = heat_vacuum_tail_bound(field, mass, wave)
    assert quadrature_change < 2e-9
    basis_values = []
    sizes = np.array([64, 128, 256, 512])
    for cutoff in sizes:
        block = RelativeBlock(field, mass, wave, int(cutoff))
        raw = block.spectral_difference(0, 0)
        estimate = block.v ** 2 / (8 * math.pi * math.sqrt(2 * field * cutoff))
        basis_values.append(raw + estimate)
    # Compare extrapolation windows rather than assigning an unjustified
    # fixed error to an empirical asymptotic fit.
    extrapolations = []
    for offset in (0, 1):
        n = sizes[offset:]
        design = np.column_stack([np.ones(len(n)), 1 / n, 1 / n ** 1.5])
        extrapolations.append(float(np.linalg.lstsq(design, basis_values[offset:], rcond=None)[0][0]))
    lower = refined - quadrature_change
    upper = refined + bound + quadrature_change
    assert min(extrapolations) <= upper and max(extrapolations) >= lower
    assert max(extrapolations) - min(extrapolations) < 1e-7
    print(f"PASS independent vacuum benchmark: heat_partial={refined:.12g}, "
          f"heat_tail<={bound:.3g}, quadrature_change={quadrature_change:.3g}, "
          f"basis_extrapolation_range={extrapolations}")
    return rows, {"B": field, "M": mass, "b": wave, "heat_partial": refined,
                  "heat_tail_bound": bound, "quadrature_change": quadrature_change,
                  "basis_extrapolations": extrapolations,
                  "warning": "Extrapolation windows are a diagnostic, not a certified basis error."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    verify_displacement()
    verify_fermi_resolvent()
    finite = verify_finite_determinants()
    convergence, benchmark = verify_basis_convergence()
    if args.output:
        from pathlib import Path
        report = {"normalization": "per unit area, per fermion copy; m_star=1",
                  "finite_matrix_checks": finite, "basis_convergence": convergence,
                  "independent_vacuum_benchmark": benchmark,
                  "limitations": "Frequency tail bound excludes oscillator truncation and floating roundoff; "
                                 "UV-tail correction/extrapolation is diagnostic, not a uniform certified bound."}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")
