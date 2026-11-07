"""Relative CDW potential in the displaced oscillator basis; see note 11.

All returned potentials are per unit area and per fermion copy. This module
requires B=|eB|>0, and uses the exact compression P_N D(alpha) P_N of the
infinite displacement operator, not exp of a truncated creation operator.
The frequency error bound does NOT bound oscillator-basis truncation.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigvalsh
from scipy.special import eval_genlaguerre, expit, gammaln, zeta


def displacement(alpha, cutoff):
    """Real <n|D(alpha)|m>; exact Laguerre formula, then compression."""
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    if alpha == 0:
        return np.eye(cutoff)
    result = np.zeros((cutoff, cutoff))
    amplitude = abs(alpha)
    for difference in range(cutoff):
        lower = np.arange(cutoff - difference)
        log_prefactor = (-amplitude ** 2 / 2 + difference * math.log(amplitude)
                         + (gammaln(lower + 1) - gammaln(lower + difference + 1)) / 2)
        values = np.exp(log_prefactor) * eval_genlaguerre(lower, difference, amplitude ** 2)
        result[lower + difference, lower] = values * np.sign(alpha) ** difference
        result[lower, lower + difference] = values * (-np.sign(alpha)) ** difference
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Displacement matrix requires a more stable evaluation at these parameters")
    return result


def fermi_resolvent(lam, chemical, temperature):
    """G(lambda)=T sum 1/[(omega-i mu)^2+lambda], and its derivative."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0):
        raise ValueError("lambda must be nonnegative")
    energy = np.sqrt(lam)
    x, y = energy / temperature, abs(chemical) / temperature
    small = x < 1e-3
    small_occupation = expit(-y)
    sech_squared = 4 * small_occupation * (1 - small_occupation)
    g_zero = sech_squared / (4 * temperature)
    derivative_zero = sech_squared * (2 - 3 * sech_squared) / (48 * temperature ** 3)
    G = np.empty_like(lam)
    derivative = np.empty_like(lam)
    G[small] = g_zero + derivative_zero * lam[small]
    derivative[small] = derivative_zero
    regular = ~small
    xr = x[regular]
    er = energy[regular]
    scale = np.maximum(xr, y)
    numerator = np.exp(xr - scale) * (-np.expm1(-2 * xr))
    denominator = (np.exp(xr - scale) + np.exp(-xr - scale)
                   + np.exp(y - scale) + np.exp(-y - scale))
    w = numerator / denominator
    sm = expit(-np.abs((er - chemical) / temperature))
    sp = expit(-np.abs((er + chemical) / temperature))
    w_prime = (sm * (1 - sm) + sp * (1 - sp)) / temperature
    G[regular] = w / (2 * er)
    derivative[regular] = w_prime / (4 * er ** 2) - w / (4 * er ** 3)
    return G, derivative


@dataclass(frozen=True)
class MatsubaraResult:
    value: float
    frequency_tail_bound: float
    positive_frequencies: int
    oscillator_cutoff: int


class RelativeBlock:
    def __init__(self, field, mass, wave, cutoff):
        if field <= 0 or cutoff < 1:
            raise ValueError("require field=|eB|>0 and cutoff>=1")
        self.field = float(field)
        self.mass = abs(float(mass))
        self.wave = abs(float(wave))
        self.cutoff = int(cutoff)
        self.v = 2 * self.mass * self.wave
        self.degeneracy = self.field / (2 * math.pi)
        n = np.arange(self.cutoff)
        self.upper = self.mass ** 2 + 2 * self.field * (n + 1)
        self.lower = self.mass ** 2 + 2 * self.field * n
        self.D = (np.eye(self.cutoff) if self.v == 0 else
                  displacement(math.sqrt(2 / self.field) * self.wave, self.cutoff))
        self.weights = self.D ** 2

    def schur_log(self, spectral_shift):
        """Return Re log det S(z), Tr K(z); S=I-K."""
        if self.v == 0:
            return 0.0, 0.0j
        shift = complex(spectral_shift)
        if shift.imag == 0 and shift.real >= 0:
            # The real-frequency Schur complement is similar to a positive
            # matrix. Cholesky checks positivity instead of hiding a bad
            # pivot behind log(abs(det)).
            rp = 1 / (self.upper + shift.real)
            rm = 1 / (self.lower + shift.real)
            weighted = np.sqrt(rp)[:, None] * self.D
            K = self.v ** 2 * (weighted * rm[None, :]) @ weighted.T
            try:
                factor = np.linalg.cholesky(np.eye(self.cutoff) - K)
            except np.linalg.LinAlgError as error:
                raise FloatingPointError("Positive Schur complement lost precision; inspect the spectral gap") from error
            log_absolute = 2 * float(np.sum(np.log(np.diag(factor))))
        else:
            rp = 1 / (self.upper + shift)
            rm = 1 / (self.lower + shift)
            K = self.v ** 2 * ((rp[:, None] * self.D) * rm[None, :]) @ self.D.T
            _, log_absolute = np.linalg.slogdet(np.eye(self.cutoff) - K)
        if not np.isfinite(log_absolute):
            raise FloatingPointError("Singular relative determinant; increase precision or inspect the spectral gap")
        return float(log_absolute), complex(np.trace(K))

    def first_frequency_term(self, chemical, temperature):
        """Exact full-frequency sum of d_B*T*Tr K for this compression."""
        if self.v == 0:
            return 0.0
        gp, derivative = fermi_resolvent(self.upper, chemical, temperature)
        gm, _ = fermi_resolvent(self.lower, chemical, temperature)
        differences = self.lower[None, :] - self.upper[:, None]
        quotient = np.zeros_like(differences)
        # Equality is known by indices (m=n+1), not by a floating threshold.
        equal = np.arange(self.cutoff)[None, :] == np.arange(self.cutoff)[:, None] + 1
        np.divide(gp[:, None] - gm[None, :], differences, out=quotient, where=~equal)
        rows, columns = np.nonzero(equal)
        quotient[rows, columns] = -derivative[rows]
        return self.degeneracy * self.v ** 2 * float(np.sum(self.weights * quotient))

    def frequency_tail_bound(self, chemical, temperature, terms):
        """Uniform-in-cutoff bound after subtracting the first frequency term."""
        if self.v == 0:
            return 0.0
        omega = (2 * terms + 1) * math.pi * temperature
        if omega ** 2 < 2 * chemical ** 2:
            return math.inf
        kappa = 4 * self.v ** 2 / omega ** 4
        if kappa >= 1:
            return math.inf
        sums = {p: float(zeta(p, terms + 0.5)) / (2 * math.pi * temperature) ** p
                for p in (6, 8)}
        return (4 * self.degeneracy * temperature * self.v ** 4 / (1 - kappa)
                * (4 * sums[8] + sums[6] / self.field))

    def matsubara(self, chemical, temperature, tolerance=1e-9, maximum_terms=16384):
        if temperature <= 0 or tolerance <= 0:
            raise ValueError("require temperature>0 and tolerance>0")
        if self.v == 0:
            return MatsubaraResult(0.0, 0.0, 0, self.cutoff)
        terms = 8
        while self.frequency_tail_bound(chemical, temperature, terms) > tolerance:
            terms *= 2
            if terms > maximum_terms:
                raise RuntimeError("Frequency bound exceeds maximum_terms; use a larger budget or higher T")
        residuals = []
        for index in range(terms):
            omega = (2 * index + 1) * math.pi * temperature
            logabs, trace = self.schur_log((omega - 1j * chemical) ** 2)
            residuals.append(logabs + trace.real)
        value = (self.first_frequency_term(chemical, temperature)
                 - 2 * self.degeneracy * temperature * math.fsum(residuals))
        return MatsubaraResult(value, self.frequency_tail_bound(chemical, temperature, terms),
                              terms, self.cutoff)

    def vacuum(self, tolerance=1e-10):
        if self.v == 0:
            return 0.0, 0.0
        integral, error = quad(lambda omega: self.schur_log(omega ** 2)[0],
                               0, np.inf, epsabs=tolerance * math.pi / self.degeneracy,
                               epsrel=tolerance, limit=250)
        return -self.degeneracy * integral / math.pi, self.degeneracy * error / math.pi

    def eigenvalues(self):
        """For independent checks; not used by matsubara() or vacuum()."""
        matrix = np.block([[np.diag(self.upper), self.v * self.D],
                           [self.v * self.D.T, np.diag(self.lower)]])
        values = eigvalsh(matrix)
        if values.min() < -1e-10:
            raise FloatingPointError("Compression lost positivity")
        return np.maximum(values, 0)

    def spectral_difference(self, chemical, temperature):
        """Independent finite-compression reference; includes vacuum difference."""
        def F(values):
            E = np.sqrt(values)
            if temperature == 0:
                return np.maximum(E, abs(chemical))
            return E + temperature * (np.logaddexp(0, (chemical - E) / temperature)
                                       + np.logaddexp(0, (-chemical - E) / temperature))

        return -self.degeneracy * (float(np.sum(F(self.eigenvalues())))
                                   - float(np.sum(F(np.r_[self.upper, self.lower]))))
