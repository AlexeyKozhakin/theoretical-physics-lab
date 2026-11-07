"""Check the B->0 heat-density limit against the independent dispersion.

Run: python3 cdw-magnetic/code/check_zero_field_limit.py
The finite-B side uses the integral series, not a truncated Landau spectrum.
The zero-field side integrates the explicit E_+/- dispersion directly.
"""

import math

import numpy as np
from scipy.integrate import quad

from check_heat_kernel import gaussian_spin_trace_term


def zero_field_density(proper, mass, wave):
    def integrand(momentum):
        root = math.hypot(momentum, mass)
        return math.exp(-proper * (root + wave) ** 2) + math.exp(-proper * (root - wave) ** 2)

    value = quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-13)[0]
    return value / (2 * math.pi ** 1.5 * math.sqrt(proper))


def finite_field_density(field, proper, mass, wave):
    t = 2 * field * proper
    coupling = wave / math.sqrt(2 * field)
    splitting = wave * mass / field
    constant = mass ** 2 + wave ** 2 + field
    prefactor = field / (2 * math.pi) * math.exp(-constant * proper)
    terms = [gaussian_spin_trace_term(t, coupling, splitting, 0.5, r,
                                     nodes=(1, 28, 12, 7)[r])
             for r in range(4)]
    bound = terms[0] * (math.cosh(splitting * t)
                        - math.fsum((splitting * t) ** (2 * r) / math.factorial(2 * r)
                                    for r in range(4)))
    return prefactor * math.fsum(terms), prefactor * bound


if __name__ == "__main__":
    # Includes gapped b<M, touching b=M, and gapless b>M zero-field spectra.
    for proper, mass, wave in ((0.6, 0.7, 0.4), (0.7, 0.5, 0.5), (0.6, 0.3, 0.8)):
        target = zero_field_density(proper, mass, wave)
        errors = []
        for field in (0.4, 0.2, 0.1, 0.05):
            value, bound = finite_field_density(field, proper, mass, wave)
            error = abs(value - target)
            errors.append(error)
            print(f"B={field:g}, s={proper:g}, M={mass:g}, b={wave:g}: "
                  f"heat_error={error:.6g}, series_tail<={bound:.3g}")
            assert error > 20 * bound
        ratios = [first / second for first, second in zip(errors, errors[1:])]
        assert all(3.5 < ratio < 4.5 for ratio in ratios), ratios
        print(f"PASS B->0: error ratios={ratios}; independent momentum integral.")
