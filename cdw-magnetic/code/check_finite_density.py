"""Checks for note 09: arbitrary chemical potential and spectral determinant.

Run: python3 cdw-magnetic/code/check_finite_density.py
Uses NumPy, SciPy, mpmath. Vacuum energy is not estimated by a finite
matrix: only the UV-finite thermal contribution is used in these tests.
"""

import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigvalsh


def spectrum(field, mass, wave, cutoff):
    a = np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), 1)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sz = np.diag([1.0, -1.0])
    h = (np.kron(np.diag(np.arange(cutoff, dtype=float)), np.eye(2))
         + wave / math.sqrt(2 * field) * np.kron(a + a.T, sz)
         + wave * mass / field * np.kron(np.eye(cutoff), sx)
         + 0.5 * np.kron(np.eye(cutoff), sz))
    eigenvalues = mass ** 2 + wave ** 2 + field + 2 * field * eigvalsh(h)
    assert eigenvalues.min() > -1e-11
    return np.sqrt(np.maximum(eigenvalues, 0))


def weight(energy, chemical, temperature):
    return temperature * (np.logaddexp(0, (chemical - energy) / temperature)
                          + np.logaddexp(0, (-chemical - energy) / temperature))


def split_thermal(energies, chemical, temperature, separation, order):
    assert separation > abs(chemical)
    low = energies[energies <= separation]
    high = energies[energies > separation]
    result = -float(np.sum(weight(low, chemical, temperature)))
    for index in range(1, order + 1):
        terms = (np.exp(-index * (high - chemical) / temperature)
                 + np.exp(-index * (high + chemical) / temperature))
        result -= temperature * (-1) ** (index + 1) / index * float(np.sum(terms))
    index = order + 1
    remainder = temperature / index * float(np.sum(
        np.exp(-index * (high - chemical) / temperature)
        + np.exp(-index * (high + chemical) / temperature)))
    return result, remainder, high


def subordinated_boltzmann(energies, inverse_temperature, separation):
    """Compute exp(a*Ec) Tr_high exp(-a sqrt(L)) from its heat trace."""
    a = inverse_temperature
    squares = energies ** 2

    def integrand(v):
        if v == 0:
            return 0.0
        exponents = -v ** 2 - a ** 2 * squares / (4 * v ** 2) + a * separation
        # All exponents are <= -a*(E-Ec) < 0 by AM-GM.
        return 2 / math.sqrt(math.pi) * float(np.sum(np.exp(exponents)))

    return quad(integrand, 0, np.inf, epsabs=2e-12, epsrel=2e-12, limit=250)[0]


def theta_product(s, high, chemical, temperature):
    """k_high(s)*(1-theta4), avoiding large intermediate exponentials."""
    if s == 0:
        return 0.0
    squares = high ** 2
    if s <= 1 / (4 * temperature ** 2):
        # Winding form, with exponents combined with k_high before exp.
        count = math.ceil(2 * temperature * abs(chemical) * s
                          + math.sqrt(4 * temperature ** 2 * s * 70)) + 3
        values = []
        for index in range(1, count + 1):
            common = -squares * s - index ** 2 / (4 * temperature ** 2 * s)
            pair = (np.exp(common + index * chemical / temperature)
                    + np.exp(common - index * chemical / temperature))
            values.append((-1) ** (index + 1) * float(np.sum(pair)))
        return math.fsum(values)
    # Poisson/Matsubara form, valid here with exponentially small omitted tail.
    first = float(np.sum(np.exp(-squares * s)))
    terms = []
    for index in range(8):
        omega = (2 * index + 1) * math.pi * temperature
        amplitudes = np.exp(-(squares + omega ** 2 - chemical ** 2) * s)
        terms.append(math.cos(2 * omega * chemical * s) * float(np.sum(amplitudes)))
    return first - 4 * temperature * math.sqrt(math.pi * s) * math.fsum(terms)


def check_thermal():
    field, mass, wave = 0.8, 0.7, 0.4
    energies = spectrum(field, mass, wave, 75)
    larger = spectrum(field, mass, wave, 105)
    for chemical, temperature in ((0.2, 0.25), (1.8, 0.2), (4.0, 0.15)):
        direct = -float(np.sum(weight(energies, chemical, temperature)))
        reference = -float(np.sum(weight(larger, chemical, temperature)))
        assert abs(direct - reference) < 2e-10
        for margin in (1.5, 3.0, 5.0):
            separation = abs(chemical) + margin * temperature
            computed, bound, high = split_thermal(
                energies, chemical, temperature, separation, order=10)
            error = abs(computed - direct)
            assert error <= bound + 2e-12
        separation = abs(chemical) + 3 * temperature
        _, _, high = split_thermal(energies, chemical, temperature, separation, order=10)
        for index in (1, 2, 5):
            a = index / temperature
            computed = subordinated_boltzmann(high, a, separation)
            target = float(np.sum(np.exp(-a * (high - separation))))
            assert abs(computed - target) < 2e-10
        print(f"PASS thermal/subordination: mu={chemical}, T={temperature}; "
              "three separation energies, finite-matrix convergence.")

    # Direct check of the compact theta integral in a case with occupied levels.
    chemical, temperature = 1.8, 0.2
    separation = chemical + 3 * temperature
    high = energies[energies > separation]
    target = -float(np.sum(weight(high, chemical, temperature)))
    switch = 1 / (4 * temperature ** 2)

    def integrand(s):
        if not s:
            return 0.0
        return theta_product(s, high, chemical, temperature) / s ** 1.5

    computed = -(quad(integrand, 0, switch, epsabs=1e-11, limit=250)[0]
                 + quad(integrand, switch, np.inf, epsabs=1e-11, limit=250)[0]) / (2 * math.sqrt(math.pi))
    assert abs(computed - target) < 2e-10
    print(f"PASS theta integral above the gap: error={abs(computed-target):.3g}")

    # An exactly massless homogeneous L contains one zero eigenvalue per block.
    zero_spectrum = spectrum(0.8, 0.0, 0.0, 75)
    assert zero_spectrum[0] == 0
    for chemical, temperature in ((0.0, 0.25), (0.7, 0.3)):
        target = -float(np.sum(weight(zero_spectrum, chemical, temperature)))
        computed, bound, _ = split_thermal(zero_spectrum, chemical, temperature,
                                          abs(chemical) + 3 * temperature, order=12)
        assert abs(computed - target) < bound + 2e-12
    print("PASS zero modes: full thermal weight, no extra double counting.")

    # T=0 limit at fixed spectrum, away from thresholds.
    chemical = 1.7
    target = -float(np.sum(np.maximum(chemical - energies, 0)))
    errors = [abs(-float(np.sum(weight(energies, chemical, temperature))) - target)
              for temperature in (0.08, 0.04, 0.02)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-4
    print(f"PASS T->0: errors={[f'{error:.3g}' for error in errors]}")


def determinant_coefficients(moments, order):
    coefficients = [mp.mpf(1), mp.mpf(0)]
    for index in range(2, order + 1):
        coefficients.append(-mp.fsum(moments[m] * coefficients[index - m]
                                     for m in range(2, index + 1)) / index)
    return coefficients


def check_determinant():
    mp.mp.dps = 90
    field, mass, shift = mp.mpf("0.8"), mp.mpf("0.7"), mp.mpf("1.0")
    nu = (mass ** 2 + shift) / (2 * field)
    order = 110
    moments = {m: (2 * mp.zeta(m, nu) - nu ** (-m)) / (2 * field) ** m
               for m in range(2, order + 1)}
    coefficients = determinant_coefficients(moments, order)
    for z in (mp.mpf("-0.4"), mp.mpf("0.2"), mp.mpf("2.4"), mp.mpc("1.1", "0.2")):
        w = (z + shift) / (2 * field)
        reference = (mp.gamma(nu) ** 2 * mp.rgamma(nu - w) ** 2
                     * mp.exp(-2 * w * mp.digamma(nu) - w / nu) / (1 - w / nu))
        computed = mp.fsum(c * (z + shift) ** n for n, c in enumerate(coefficients))
        error = abs(computed - reference)
        assert error < mp.mpf("1e-35"), (z, error)
        print(f"PASS entire determinant: z={z}, error={float(error):.3g}")

    for m in (2, 3, 4, 8):
        def integrand(s):
            if s == 0:
                return 1 / field if m == 2 else mp.mpf(0)
            return s ** (m - 1) * mp.exp(-(shift + mass ** 2) * s) * mp.coth(field * s)
        from_heat = mp.quad(integrand, [0, 1, mp.inf]) / mp.gamma(m)
        assert abs(from_heat - moments[m]) < mp.mpf("1e-60")
    print("PASS determinant moments: independently integrated homogeneous heat trace.")


def check_contour_prefix():
    """Residue formula: double positive roots and a separately counted zero."""
    mp.mp.dps = 45
    field, shift = mp.mpf("0.8"), mp.mpf("1.0")
    chemical, temperature = mp.mpf("1.2"), mp.mpf("0.2")
    left, right, height = mp.mpf("0.15"), mp.mpf("1.7"), mp.mpf("0.06")
    corners = [mp.mpc(left, -height), mp.mpc(right, -height),
               mp.mpc(right, height), mp.mpc(left, height)]
    for mass in (mp.mpf("0.7"), mp.mpf(0)):
        nu = (mass ** 2 + shift) / (2 * field)

        def logarithmic_derivative(energy):
            w = (energy ** 2 + shift) / (2 * field)
            derivative_lambda = (2 * mp.digamma(nu - w) - 2 * mp.digamma(nu)
                                 - 1 / nu + 1 / (nu - w)) / (2 * field)
            return 2 * energy * derivative_lambda

        functions = (
            lambda energy: mp.exp(-mp.mpf("0.7") * energy ** 2),
            lambda energy: temperature * (
                mp.log(1 + mp.exp(-(energy - chemical) / temperature))
                + mp.log(1 + mp.exp(-(energy + chemical) / temperature))))
        for index, function in enumerate(functions):
            integral = mp.mpc(0)
            for start, end in zip(corners, corners[1:] + corners[:1]):
                integral += mp.quad(
                    lambda u: function(start + (end - start) * u)
                    * logarithmic_derivative(start + (end - start) * u) * (end - start),
                    [0, 1])
            result = integral / (2 * mp.pi * 1j)
            if mass == 0:
                result += function(0)
            target = function(mass) + 2 * function(mp.sqrt(mass ** 2 + 2 * field))
            error = abs(result - target)
            assert error < mp.mpf("1e-18"), (mass, index, error)
            print(f"PASS contour prefix: M={mass}, weight={index}, error={float(error):.3g}")


if __name__ == "__main__":
    check_thermal()
    check_determinant()
    check_contour_prefix()
