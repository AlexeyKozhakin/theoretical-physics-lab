"""Independent checks for the Gaussian spin-flip heat trace in note 08.

Run: python3 cdw-magnetic/code/check_heat_kernel.py
Dependencies: NumPy, SciPy, mpmath, SymPy (versions printed).
Finite oscillator matrices are used only as a numerical reference at fixed
positive proper time, with a cutoff convergence check, not to infer UV terms.
"""

import math

import mpmath as mp
import numpy as np
import scipy
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigvalsh


def ordered_quadrature(dimension, nodes):
    """Gauss rule on 0 < u1 < ... < ud < 1, including its Jacobian."""
    points, weights = leggauss(nodes)
    points, weights = (points + 1) / 2, weights / 2
    grids = np.meshgrid(*([np.arange(nodes)] * dimension), indexing="ij")
    indices = np.stack([grid.ravel() for grid in grids], axis=-1)
    cube = points[indices]
    quadrature_weights = np.prod(weights[indices], axis=1)
    previous = np.zeros(len(cube))
    times = np.empty_like(cube)
    jacobian = np.ones(len(cube))
    for index in range(dimension):
        jacobian *= 1 - previous
        previous = previous + (1 - previous) * cube[:, index]
        times[:, index] = previous
    return times, quadrature_weights * jacobian


def gaussian_spin_trace_term(t, coupling, splitting, bias, flips, nodes=8):
    """Term with 2*flips spin changes in the exact series for Z_R(t)."""
    base = 2 * math.exp(coupling ** 2 * t) * math.cosh(bias * t) / (-math.expm1(-t))
    if flips == 0 or splitting == 0:
        return base if flips == 0 else 0.0
    dimension = 2 * flips
    u, weights = ordered_quadrature(dimension, nodes)
    signs = (-1.0) ** np.arange(dimension)
    ell = 1 + 2 * (u @ signs)
    # J=t+4 sum_{i<j} (-1)^(i+j) R_t(t|u_i-u_j|).
    # expm1 avoids subtracting two O(1/t) covariance functions.
    J = np.full(len(u), t)
    for first in range(dimension):
        for second in range(first + 1, dimension):
            distance = t * (u[:, second] - u[:, first])
            R = ((-np.expm1(-distance)) * (-np.expm1(-(t - distance)))
                 / (-np.expm1(-t)))
            J += 4 * signs[first] * signs[second] * R
    assert np.min(J) >= -2e-12 and np.max(J) <= t + 2e-12
    normalized_integrand = (np.exp(coupling ** 2 * (J - t))
                            * np.cosh(bias * t * ell) / math.cosh(bias * t))
    assert np.min(normalized_integrand) > 0
    assert np.max(normalized_integrand) <= 1 + 2e-12
    # Integral on the unit ordered simplex; no extra factorial belongs here.
    return base * (splitting * t) ** dimension * float(weights @ normalized_integrand)


def diagonal_trace(t, coupling, splitting, bias, cutoff):
    n = np.diag(np.arange(cutoff, dtype=float))
    a = np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), 1)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.diag([1.0, -1.0])
    h = (np.kron(n, np.eye(2)) + coupling * np.kron(a + a.T, sz)
         + splitting * np.kron(np.eye(cutoff), sx)
         + bias * np.kron(np.eye(cutoff), sz))
    return float(np.sum(np.exp(-t * eigvalsh(h))))


def check_series():
    cases = ((0.7, 0.35, 0.4, 0.5),
             (1.1, 0.8, 0.25, 0.5),
             (0.9, 0.0, 0.5, 0.5))
    for t, coupling, splitting, bias in cases:
        terms = [gaussian_spin_trace_term(t, coupling, splitting, bias, r,
                                         nodes=(1, 24, 12, 7)[r])
                 for r in range(4)]
        approximation = math.fsum(terms)
        reference = diagonal_trace(t, coupling, splitting, bias, 70)
        larger_reference = diagonal_trace(t, coupling, splitting, bias, 100)
        assert abs(reference - larger_reference) < 2e-12
        z0 = terms[0]
        tail = z0 * (math.cosh(splitting * t)
                     - math.fsum((splitting * t) ** (2 * r)
                                 / math.factorial(2 * r) for r in range(4)))
        # Quadrature convergence is checked independently of the truncation tail.
        refined = math.fsum(gaussian_spin_trace_term(t, coupling, splitting, bias, r,
                                                   nodes=(1, 32, 16, 8)[r])
                           for r in range(4))
        quadrature_error = abs(refined - approximation)
        assert quadrature_error < 1e-10
        error = larger_reference - refined
        assert -1e-10 <= error <= tail + 1e-10
        if coupling == 0:
            exact = 2 * math.cosh(t * math.hypot(splitting, bias)) / (-math.expm1(-t))
            assert abs(exact - larger_reference) < 2e-12
        print(f"PASS series: t={t}, g={coupling}, Delta={splitting}; "
              f"error={error:.3g}, tail_bound={tail:.3g}, "
              f"quadrature_change={quadrature_error:.3g}")


def check_uv():
    field, mass, wave = 0.8, 0.7, 0.6
    target = 2 * wave ** 2 * mass ** 2 / field
    errors = []
    for proper in (1e-3, 1e-4, 1e-5):
        t = 2 * field * proper
        coupling = wave / math.sqrt(2 * field)
        splitting = wave * mass / field
        constant = mass ** 2 + wave ** 2 + field
        correction = math.exp(-constant * proper) * gaussian_spin_trace_term(
            t, coupling, splitting, 0.5, 1, nodes=32)
        # Higher terms start at O(s^3); no subtraction of two divergent traces.
        error = abs(correction / proper - target)
        errors.append(error)
        print(f"PASS UV sample: s={proper:g}, (k_b-k_0)/s={correction/proper:.10g}, "
              f"target={target:.10g}")
    assert errors[1] < errors[0] / 8 and errors[2] < errors[1] / 8
    assert errors[-1] < 5e-6

    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.diag(1, -1)
    I2, I4 = sp.eye(2), sp.eye(4)
    B, M, b = sp.symbols("B M b", real=True)
    S = sp.kronecker_product(I2, sz)
    C = -sp.kronecker_product(sx, sx)
    A = -sp.kronecker_product(sy, I2)
    potential = M ** 2 * I4 + B * S + 2 * b * M * C
    curvature = sp.I * B * I4
    coefficient = ((potential ** 2).trace() / 2
                   + (2 * curvature ** 2).trace() / 12)
    assert sp.simplify(coefficient - (2 * M ** 4 + 4 * B ** 2 / 3 + 8 * b ** 2 * M ** 2)) == 0
    # Trace of the covariant Laplacian of the endomorphism vanishes.
    derivative = sp.I * b * (A * potential - potential * A)
    second_derivative = sp.I * b * (A * derivative - derivative * A)
    assert sp.simplify(second_derivative.trace()) == 0
    print("PASS UV: independent local heat-coefficient calculation.")


def homogeneous_integral(field, mass):
    """Renormalized proper-time integral, omitting M^2/(2g)."""
    B, M = mp.mpf(str(field)), mp.mpf(str(mass))

    def integrand(u):
        if not u:
            return M ** 4 / B + 2 * B / 3
        s = u ** 2
        z, x = M ** 2 * s, B * s
        if abs(z) < mp.mpf("0.01"):
            exp_remainder = mp.fsum((-z) ** k / mp.factorial(k) for k in range(2, 20))
        else:
            exp_remainder = mp.expm1(-z) + z
        if abs(x) < mp.mpf("0.01"):
            coth_remainder = (x / 3 - x ** 3 / 45 + 2 * x ** 5 / 945
                              - x ** 7 / 4725 + 2 * x ** 9 / 93555)
        else:
            coth_remainder = mp.coth(x) - 1 / x
        subtracted = exp_remainder / (B * s) + mp.exp(-z) * coth_remainder
        return 2 * subtracted / u ** 2

    return B / (4 * mp.pi ** mp.mpf("1.5")) * mp.quad(integrand, [0, 1, mp.inf])


def check_vacuum_normalization():
    mp.mp.dps = 40
    for field, mass in ((0.8, 0.7), (1.2, 0.0), (0.25, 1.1)):
        computed = homogeneous_integral(field, mass)
        B, M = mp.mpf(str(field)), mp.mpf(str(mass))
        zeta = (M * B / (2 * mp.pi)
                - (2 * B) ** mp.mpf("1.5") / (2 * mp.pi)
                * mp.zeta(-mp.mpf("0.5"), M ** 2 / (2 * B)))
        error = abs(computed - zeta)
        assert error < mp.mpf("1e-20"), (field, mass, error)
        print(f"PASS vacuum/zeta: B={field}, M={mass}, error={float(error):.3g}")


if __name__ == "__main__":
    print(f"NumPy {np.__version__}; SciPy {scipy.__version__}; "
          f"mpmath {mp.__version__}; SymPy {sp.__version__}")
    check_series()
    check_uv()
    check_vacuum_normalization()
