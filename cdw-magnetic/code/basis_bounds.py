"""Analytic oscillator-basis bounds with explicitly checked applicability.

The vacuum upper bound uses either global kappa0<1 or the discarded-tail
and boundary Schur comparison. The latter can cover strong mixing.
Analytic inequalities are distinct from numerical estimates of quadrature
and floating error. No interval-arithmetic certification is claimed.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.linalg import eigh, eigvalsh
from scipy.special import zeta

from homogeneous_potential import ladder_thermal_sum, thermal_weight


@dataclass(frozen=True)
class BornVacuum:
    value: float
    quadrature_error_estimate: float


@lru_cache(maxsize=128)
def infinite_first_vacuum(field, mass, wave, tolerance=2e-11):
    """Infinite-basis first vacuum term, computed independently of D_N."""
    B, M, b = float(field), abs(float(mass)), abs(float(wave))
    if B <= 0 or tolerance <= 0:
        raise ValueError("require B>0, tolerance>0")
    if M == 0 or b == 0:
        return BornVacuum(0.0, 0.0)
    v = 2 * b * M
    alpha_squared = 2 * b ** 2 / B
    inner_accuracy = 2e-12
    max_inner_error = 0.0

    def I(proper):
        nonlocal max_inner_error
        t = 2 * B * proper
        if t == 0:
            return 1.0
        if t < 2:
            def kernel(y):
                R = (-math.expm1(-t * y)) * (-math.expm1(-t * (1 - y))) / (-math.expm1(-t))
                ratio = (math.exp(-t * y) + math.exp(-t * (1 - y))) / (1 + math.exp(-t))
                return 2 * math.exp(-alpha_squared * R) * ratio
            result, error = quad(kernel, 0, 0.5, epsabs=inner_accuracy, epsrel=inner_accuracy)
        else:
            end = min(t / 2, 48.0)
            factor = 2 / (t * (1 + math.exp(-t)))

            def kernel(q):
                R = (-math.expm1(-q)) * (-math.expm1(-(t - q))) / (-math.expm1(-t))
                return (math.exp(-q) + math.exp(-(t - q))) * math.exp(-alpha_squared * R)

            integral, estimate = quad(kernel, 0, end, epsabs=inner_accuracy,
                                      epsrel=inner_accuracy)
            result = factor * integral
            omitted = (factor * (math.exp(-end) - math.exp(-(t - end))) if end < t / 2 else 0)
            error = factor * estimate + omitted
        max_inner_error = max(max_inner_error, error)
        return result

    # Remove the known large-s term analytically. Otherwise small M creates
    # two separated scales and a quadrature can miss a narrow O(M^2) correction.
    A = -math.expm1(-alpha_squared) / alpha_squared
    leading = B * M / (4 * math.pi) * (-math.expm1(-alpha_squared))
    ratio = M ** 2 / B
    limit = 8.0

    def integrand(u):
        if u == 0:
            return 1 - A
        bracket = u ** 2 / math.tanh(u ** 2) * I(u ** 2 / B) - A
        return math.exp(-ratio * u ** 2) * bracket

    split = min(limit, 1 / math.sqrt(ratio))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            raw, error = quad(integrand, 0, split, epsabs=tolerance, epsrel=tolerance, limit=200)
            rest, rest_error = quad(integrand, split, limit, epsabs=tolerance, epsrel=tolerance, limit=200)
    except IntegrationWarning as error:
        raise FloatingPointError("First vacuum integral did not meet its quadrature checks") from error

    def J0(a):
        return math.sqrt(math.pi) / (2 * math.sqrt(a)) * math.erfc(math.sqrt(a) * limit)

    def J2(a):
        return (limit * math.exp(-a * limit ** 2) / (2 * a)
                + math.sqrt(math.pi) / (4 * a ** 1.5) * math.erfc(math.sqrt(a) * limit))

    denominator = -math.expm1(-2 * limit ** 2)
    analytic_tail = ((2 * J0(ratio + 1) + J0(ratio + 2)) / denominator
                     + alpha_squared * J2(ratio + 2) / denominator ** 2)
    weight_bound = min(limit ** 3 / 3 + limit,
                       math.sqrt(math.pi) / (4 * ratio ** 1.5)
                       + math.sqrt(math.pi) / (2 * math.sqrt(ratio)))
    prefactor = B / (2 * math.pi) * 2 * b ** 2 * M ** 2 / (math.sqrt(math.pi) * B ** 1.5)
    estimate = prefactor * (error + rest_error + analytic_tail + max_inner_error * weight_bound)
    return BornVacuum(leading + prefactor * (raw + rest), estimate)


def finite_first_vacuum(block):
    if block.v == 0:
        return 0.0
    plus, minus = np.sqrt(block.upper), np.sqrt(block.lower)
    quotient = 1 / (2 * plus[:, None] * minus[None, :] * (plus[:, None] + minus[None, :]))
    return block.degeneracy * block.v ** 2 * float(np.sum(block.weights * quotient))


@dataclass(frozen=True)
class VacuumBasisBound:
    supported: bool
    lower: float
    upper: float
    kappa0: float
    first_term_tail: float
    born_quadrature_error_estimate: float
    upper_method: str = "global_kappa"
    tail_block_upper: float | None = None
    boundary_upper: float | None = None
    discarded_schur_lower: float | None = None


def compressed_residual(block):
    N, v = block.cutoff, abs(block.v)
    matrix = np.block([[np.diag(block.upper), block.v * block.D],
                       [block.v * block.D.T, np.diag(block.lower)]])
    residual = v ** 2 * np.block([[np.eye(N) - block.D @ block.D.T, np.zeros((N, N))],
                                  [np.zeros((N, N)), np.eye(N) - block.D.T @ block.D]])
    return matrix, (residual + residual.T) / 2


def tail_schur_vacuum_upper(block):
    """Upper bound from the discarded tail and its finite-rank boundary.

    Requires gamma_N>0 and a positive finite Schur lower comparison.
    The returned margin is numerical, not interval-arithmetic certification.
    """
    N, B, M, v = block.cutoff, block.field, block.mass, abs(block.v)
    gamma = M ** 2 + 2 * B * N - v
    if gamma <= 0:
        return None
    matrix, residual = compressed_residual(block)
    r, U = eigh(residual)
    if r[0] < -1e-10 * max(1.0, v ** 2):
        raise FloatingPointError("Residual Gram matrix lost positivity")
    # Project tiny negative floating eigenvalues in the conservative direction.
    r = np.maximum(r, 0)
    positive_residual = (U * r) @ U.T
    schur_lower = float(eigvalsh(matrix - positive_residual / gamma, subset_by_index=[0, 0])[0])
    guard_scale = max(1.0, float(np.max(block.upper)))
    if schur_lower <= 5e-13 * guard_scale:
        return None
    square_root = (U * np.sqrt(r)) @ U.T
    auxiliary = np.block([[matrix, square_root], [square_root, gamma * np.eye(2 * N)]])
    auxiliary_values = eigvalsh(auxiliary)
    if auxiliary_values[0] < -5e-12 * guard_scale:
        raise FloatingPointError("Auxiliary Schur comparison lost positivity")
    roots_original = float(np.sum(np.sqrt(np.maximum(eigvalsh(matrix), 0))))
    roots_auxiliary = float(np.sum(np.sqrt(np.maximum(auxiliary_values, 0))))
    numerical_guard = 5e-12 * (1 + block.degeneracy * (roots_original + roots_auxiliary
                                                    + 2 * N * math.sqrt(gamma)))
    boundary = block.degeneracy * (roots_original + 2 * N * math.sqrt(gamma) - roots_auxiliary)
    if boundary < -numerical_guard:
        raise FloatingPointError("Boundary vacuum upper bound became negative")
    boundary = max(0.0, boundary) + numerical_guard
    tail_upper = M ** 2 + 2 * B * (N + 1)
    tail_lower = M ** 2 + 2 * B * N
    kappa_tail = v ** 2 / (tail_upper * tail_lower)
    if kappa_tail >= 1:
        return None
    q = M ** 2 / (2 * B)
    tail = (block.degeneracy * v ** 2 / (8 * (1 - kappa_tail) * (2 * B) ** 1.5)
            * (float(zeta(1.5, N + 1 + q)) + float(zeta(1.5, N + q))))
    return {"upper": tail + boundary, "tail": tail, "boundary": boundary,
            "schur_lower": schur_lower, "numerical_guard": numerical_guard}


def vacuum_basis_bound(block):
    if block.v == 0:
        return VacuumBasisBound(True, 0.0, 0.0, 0.0, 0.0, 0.0)
    kappa = float(block.v ** 2 / (block.upper[0] * block.lower[0]))
    infinite = infinite_first_vacuum(block.field, block.mass, block.wave)
    tail = infinite.value - finite_first_vacuum(block)
    estimate = infinite.quadrature_error_estimate
    if tail < -estimate - 1e-12:
        raise ArithmeticError("First vacuum tail became negative; inspect precision or displacement matrix")
    lower = max(0.0, tail - estimate)
    upper = max(0.0, tail + estimate) / (1 - kappa) if kappa < 1 else math.inf
    method = "global_kappa" if kappa < 1 else "no_upper_bound"
    details = tail_schur_vacuum_upper(block)
    if details is not None and details["upper"] < upper:
        upper = details["upper"]
        method = "discarded_tail_schur"
    if upper < lower - estimate - 1e-10:
        raise ArithmeticError("Vacuum lower and upper basis bounds are inconsistent")
    return VacuumBasisBound(math.isfinite(upper), lower, max(lower, upper), kappa, tail, estimate,
                            method, details["tail"] if details else None,
                            details["boundary"] if details else None,
                            details["schur_lower"] if details else None)


@dataclass(frozen=True)
class ThermalBasisBound:
    lower_correction: float
    upper_correction: float
    low_levels_enclosed: int
    discarded_block_lower: float
    low_energy_cutoff_squared: float
    comparison_tail_bound: float


def thermal_basis_bound(block, chemical, temperature):
    if block.v == 0:
        return ThermalBasisBound(0.0, 0.0, 0, math.inf, math.inf, 0.0)
    N, v = block.cutoff, abs(block.v)
    gamma = block.mass ** 2 + 2 * block.field * N - v
    if gamma <= 0:
        raise ValueError("Discarded block has no positive lower bound; increase cutoff")
    threshold = gamma / 2
    matrix, residual = compressed_residual(block)
    upper_values = eigvalsh(matrix)
    if upper_values[0] < -1e-10:
        raise FloatingPointError("Compressed operator lost positivity")
    upper_values = np.maximum(upper_values, 0)
    lower_values = eigvalsh(matrix - residual / (gamma - threshold))
    lower_values = np.maximum(lower_values, 0)
    count = int(np.count_nonzero(upper_values <= threshold))
    lower_sum = float(np.sum(thermal_weight(np.sqrt(upper_values), chemical, temperature)))
    upper_low_sum = float(np.sum(thermal_weight(np.sqrt(lower_values[:count]), chemical, temperature)))
    remainder = ladder_thermal_sum(block.field, block.mass ** 2 - v, chemical, temperature,
                                   start_rank=count)
    upper_sum = upper_low_sum + remainder.upper
    if upper_sum < lower_sum - 1e-10:
        raise ArithmeticError("Thermal spectral bounds are inconsistent")
    upper_sum = max(upper_sum, lower_sum)
    # Convert S_b(infinity)-S_b(N) to the relative thermal correction;
    # the reference homogeneous trace has its own positive missing tail.
    reference = ladder_thermal_sum(block.field, block.mass ** 2, chemical, temperature,
                                   start_rank=2 * N)
    minimum = block.degeneracy * (reference.lower - (upper_sum - lower_sum))
    maximum = block.degeneracy * reference.upper
    return ThermalBasisBound(minimum, maximum, count, gamma, threshold,
                              block.degeneracy * (remainder.upper - remainder.lower
                                                  + reference.upper - reference.lower))


@dataclass(frozen=True)
class RelativeInterval:
    lower: float
    upper: float
    finite_basis_value: float
    frequency_error_bound: float
    vacuum: VacuumBasisBound
    thermal: ThermalBasisBound
    fixed_background_sign: str
    decision_tolerance: float
    numerical_status: str = "Analytic inequalities with estimated quadrature; no interval-arithmetic roundoff proof"


def relative_interval(block, chemical, temperature, frequency_tolerance=1e-9, decision_tolerance=1e-10):
    """Bound the infinite-basis relative potential at a FIXED background."""
    if decision_tolerance < 0:
        raise ValueError("decision_tolerance must be nonnegative")
    finite = block.matsubara(chemical, temperature, tolerance=frequency_tolerance)
    vac = vacuum_basis_bound(block)
    thermal = thermal_basis_bound(block, chemical, temperature)
    lower = finite.value - finite.frequency_tail_bound + vac.lower + thermal.lower_correction
    upper = finite.value + finite.frequency_tail_bound + vac.upper + thermal.upper_correction
    if block.v == 0:
        sign = "exact_zero"
    elif lower > decision_tolerance:
        sign = "positive_cost_at_fixed_mass"
    elif upper < -decision_tolerance:
        sign = "negative_cost_at_fixed_mass"
    else:
        sign = "unresolved"
    return RelativeInterval(lower, upper, finite.value, finite.frequency_tail_bound, vac, thermal, sign,
                            decision_tolerance)
