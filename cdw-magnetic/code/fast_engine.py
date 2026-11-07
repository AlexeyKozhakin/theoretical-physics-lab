"""Fast evaluation of the CDW potential Omega(M, b; mu, B, T) for phase maps.

Same physics and normalization as relative_potential.py / refine_cdw_candidate.py
(per unit area and fermion copy, g = -pi unless stated), split as

    Omega(M, b) = Omega_hom(M) + dV(M, b; B) - d_B * [sum_k H(E_k) - sum_ref H(E_ref)],

    dV = -d_B * (sum E - sum E_ref)_N + (first vacuum tail), independent of mu, T.

* dV is computed with the existing RelativeBlock + first-tail correction on a
  coarse (M, b) grid once per field and interpolated by a bicubic spline.
* Only levels with E < E_max matter for the thermal part. They are taken from
  the banded asymmetric-Rabi form of H^2 (bandwidth 2, eig_banded with a value
  window), which costs O(N) instead of the O(N^3) dense diagonalization.
* All chemical potentials of a field row reuse the same levels.

This is a search engine: values are not certified. Confirmation of a CDW point
still goes through basis_bounds.relative_interval (see phase_scout.check_candidate).
"""

from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.linalg import eig_banded

from basis_bounds import finite_first_vacuum, infinite_first_vacuum
from relative_potential import RelativeBlock


def thermal_weight(energy, chemical, temperature):
    """H_{T,mu}(E) = T ln(1+e^{-(E-mu)/T}) + T ln(1+e^{-(E+mu)/T}); T > 0."""
    E = np.asarray(energy, dtype=float)
    return temperature * (np.logaddexp(0, (chemical - E) / temperature)
                          + np.logaddexp(0, (-chemical - E) / temperature))


def rabi_band(field, mass, wave, n_osc):
    """Upper band storage (3, 2 n_osc) of h; index 2n+s, s=0,1 (sigma_z=+1,-1).

    h = a^dag a + lam (a + a^dag) sigma_z + Delta sigma_x + sigma_z / 2,
    lam = b / sqrt(2B), Delta = b M / B, and E^2 = M^2 + b^2 + B + 2 B eig(h)
    (see check_finite_density.spectrum and note 07).
    """
    lam = wave / math.sqrt(2 * field)
    delta = wave * mass / field
    size = 2 * n_osc
    band = np.zeros((3, size))
    n = np.arange(n_osc)
    band[2, 0::2] = n + 0.5
    band[2, 1::2] = n - 0.5
    band[1, 1::2] = delta                      # a[2n, 2n+1]
    couple = lam * np.sqrt(np.arange(1, n_osc))
    band[0, 2::2] = couple                     # a[2n, 2n+2], sigma_z = +1
    band[0, 3::2] = -couple                    # a[2n+1, 2n+3], sigma_z = -1
    return band


def oscillator_cutoff(field, wave, energy_max):
    """Oscillator states needed for all levels below energy_max (with margin)."""
    lam2 = wave ** 2 / (2 * field)
    top = energy_max ** 2 / (2 * field)
    return int(top + 4 * lam2 + 10 * math.sqrt(top + lam2 + 1) + 40)


def low_levels(field, mass, wave, energy_max, n_osc=None):
    """All E_k < energy_max of one block (multiplicities as in RelativeBlock)."""
    if n_osc is None:
        n_osc = oscillator_cutoff(field, wave, energy_max)
    offset = mass ** 2 + wave ** 2 + field
    # The full banded solve (O(N^2), ~0.5 ms at N ~ 100) is faster here than
    # the value-window variant; only levels below energy_max are kept.
    values = eig_banded(rabi_band(field, mass, wave, n_osc), lower=False, eigvals_only=True,
                        check_finite=False)
    squares = offset + 2 * field * values
    energies = np.sqrt(np.maximum(squares, 0.0))
    return energies[energies < energy_max]


def homogeneous_vacuum(field, mass, coupling=-math.pi):
    """Renormalized homogeneous vacuum, one multiplet (= homogeneous_potential.vacuum)."""
    with mp.workdps(30):
        B, m = mp.mpf(field), mp.mpf(abs(mass))
        z = mp.zeta(-mp.mpf("0.5"), m ** 2 / (2 * B))
        return float(m ** 2 / (2 * coupling) + m * B / (2 * mp.pi)
                     - (2 * B) ** mp.mpf("1.5") * z / (2 * mp.pi))


def vacuum_cutoff(field, wave):
    """Displaced-basis size for dV: N >= 8 kappa + 32, kappa = 2 b^2 / B (capped at 512).

    N = 64 is not enough at large kappa: at B = 0.2, b = 1.6 the error is ~2e-5,
    at B = 0.05, b = 1.6 it is ~5e-3 (checked against N = 512).
    """
    need = 8 * 2 * wave ** 2 / field + 32
    for N in (64, 128, 256, 512):
        if N >= need:
            return N
    return 512


def vacuum_difference(field, mass, wave, cutoff=None):
    """dV(M, b; B): relative vacuum at cutoff N plus the infinite first-tail correction."""
    if mass == 0 or wave == 0:
        return 0.0
    if cutoff is None:
        cutoff = vacuum_cutoff(field, wave)
    block = RelativeBlock(field, mass, wave, cutoff)
    E = np.sqrt(block.eigenvalues())
    reference = np.sqrt(np.r_[block.upper, block.lower])
    finite = -block.degeneracy * float(np.sum(E) - np.sum(reference))
    tail = infinite_first_vacuum(field, mass, wave).value - finite_first_vacuum(block)
    return finite + tail


def _vacuum_task(args):
    return vacuum_difference(*args)


class FieldRow:
    """Everything that depends on B only; mu and T enter via thermal sums."""

    def __init__(self, field, masses, waves, energy_max, vacuum_masses=None,
                 vacuum_waves=None, vacuum_cutoff=None, coupling=-math.pi, workers=1,
                 cache_dir=None):
        self.B = float(field)
        self.d = self.B / (2 * math.pi)
        self.coupling = coupling
        self.masses = np.asarray(masses, float)
        self.waves = np.asarray(waves, float)
        self.energy_max = float(energy_max)
        vm = (np.unique(np.r_[self.masses[::2], self.masses[-1]]) if vacuum_masses is None
              else np.asarray(vacuum_masses, float))
        vb = (np.unique(np.r_[self.waves[::2], self.waves[-1]]) if vacuum_waves is None
              else np.asarray(vacuum_waves, float))
        if vm[-1] < self.masses[-1] or vb[-1] < self.waves[-1]:
            raise ValueError("vacuum grid must cover the search grid")
        start = time.perf_counter()
        table = None
        cache = None
        if cache_dir is not None:
            # dV depends only on (B, M, b) and the cutoff rule: reusable for any mu, T.
            key = f"dV_B{self.B:.6g}_M{vm[0]:.4g}-{vm[-1]:.4g}x{len(vm)}_b{vb[0]:.4g}-{vb[-1]:.4g}x{len(vb)}_N{vacuum_cutoff or 'auto'}.npy"
            cache = Path(cache_dir) / key
            if cache.exists():
                table = np.load(cache)
        if table is None:
            args = [(self.B, M, b, vacuum_cutoff) for M in vm for b in vb]
            if workers > 1:
                with ProcessPoolExecutor(workers) as pool:
                    flat = list(pool.map(_vacuum_task, args, chunksize=8))
            else:
                flat = [_vacuum_task(a) for a in args]
            table = np.array(flat).reshape(len(vm), len(vb))
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, table)
        self.timing = {"vacuum_table": time.perf_counter() - start}
        self.dV = RectBivariateSpline(vm, vb, table, kx=3, ky=3, s=0)
        hm = np.linspace(0.0, max(4.0, self.masses[-1]), 801)
        self.hom_vacuum = CubicSpline(hm, [homogeneous_vacuum(self.B, M, coupling) for M in hm])
        # Levels on the search grid, padded with +inf (cached: independent of mu, T).
        start = time.perf_counter()
        level_cache = None
        if cache_dir is not None:
            level_cache = Path(cache_dir) / (
                f"levels_B{self.B:.6g}_M{self.masses[-1]:.4g}x{len(self.masses)}"
                f"_b{self.waves[-1]:.4g}x{len(self.waves)}_E{self.energy_max:.4g}.npy")
        if level_cache is not None and level_cache.exists():
            self.levels = np.load(level_cache)
        else:
            rows = [[low_levels(self.B, M, b, self.energy_max) for b in self.waves]
                    for M in self.masses]
            width = max(len(levels) for row in rows for levels in row)
            self.levels = np.full((len(self.masses), len(self.waves), width), np.inf)
            for i, row in enumerate(rows):
                for j, levels in enumerate(row):
                    self.levels[i, j, :len(levels)] = levels
            if level_cache is not None:
                np.save(level_cache, self.levels)
        self.timing["levels"] = time.perf_counter() - start
        self.vacuum_grid = (self.hom_vacuum(self.masses)[:, None]
                            + self.dV(self.masses, self.waves))

    def grid_potential(self, chemical, temperature):
        """Omega on the (M, b) search grid; levels above energy_max are dropped."""
        H = np.where(np.isfinite(self.levels),
                     thermal_weight(np.where(np.isfinite(self.levels), self.levels, 0.0),
                                    chemical, temperature), 0.0)
        return self.vacuum_grid - self.d * H.sum(axis=2)

    def potential(self, mass, wave, chemical, temperature):
        """Omega(M, b) off the grid: spline vacuum + banded levels."""
        mass, wave = abs(float(mass)), abs(float(wave))
        levels = low_levels(self.B, mass, wave, self.energy_max)
        return (float(self.hom_vacuum(mass)) + float(self.dV(mass, wave)[0, 0])
                - self.d * float(np.sum(thermal_weight(levels, chemical, temperature))))

    def direct(self, mass, wave, chemical, temperature, cutoff=None):
        """Omega(M, b) without the dV spline: RelativeBlock vacuum at the given cutoff."""
        levels = low_levels(self.B, mass, wave, self.energy_max)
        return (homogeneous_vacuum(self.B, mass, self.coupling)
                + vacuum_difference(self.B, mass, wave, cutoff)
                - self.d * float(np.sum(thermal_weight(levels, chemical, temperature))))

    def homogeneous(self, mass, chemical, temperature):
        """Homogeneous Omega(M); mass may be an array."""
        mass = np.asarray(mass, float)
        n_max = math.ceil(self.energy_max ** 2 / (2 * self.B)) + 1
        n = np.arange(n_max + 1)
        E = np.sqrt(mass[..., None] ** 2 + 2 * self.B * n)
        weights = np.r_[1.0, np.full(n_max, 2.0)]
        value = self.hom_vacuum(mass) - self.d * np.sum(
            weights * thermal_weight(E, chemical, temperature), axis=-1)
        return float(value) if value.ndim == 0 else value


def scan_row(field, chemicals, temperature, masses=None, waves=None, refine_below=5e-5,
             workers=1, cache_dir=None, confirm_margin=1e-8, confirm_below=-1e-6,
             energy_max=None):
    """Global grid search over (M, b) for every mu of one field.

    Local refinement runs only where the grid gap is below refine_below.
    """
    from scipy.optimize import minimize, minimize_scalar
    masses = np.linspace(0, 1.6, 65) if masses is None else masses
    waves = np.linspace(0, 1.7, 69) if waves is None else waves
    if energy_max is None:
        energy_max = max(chemicals) + 40 * temperature
    row = FieldRow(field, masses, waves, energy_max, workers=workers, cache_dir=cache_dir)
    start = time.perf_counter()
    hm = np.linspace(0, 4, 801)
    out = []
    for mu in chemicals:
        # homogeneous reference: grid + refinement around every grid minimum
        hv = row.homogeneous(hm, mu, temperature)
        cands = [(0.0, hv[0])]
        for i in range(1, len(hm) - 1):
            if hv[i] <= hv[i - 1] and hv[i] <= hv[i + 1]:
                r = minimize_scalar(lambda M: row.homogeneous(M, mu, temperature),
                                    bounds=(hm[i - 1], hm[i + 1]), method="bounded",
                                    options={"xatol": 1e-9})
                cands.append((float(r.x), float(r.fun)))
        M_hom, O_hom = min(cands, key=lambda c: c[1])
        grid = row.grid_potential(mu, temperature)
        grid[:, 0] = np.inf                      # b = 0 is the homogeneous family
        grid[0, :] = np.inf                      # M = 0: b irrelevant
        i, j = np.unravel_index(np.argmin(grid), grid.shape)
        best = (float(masses[i]), float(waves[j]), float(grid[i, j]))
        refined = best[2] - O_hom < refine_below
        if refined:
            res = minimize(lambda x: row.potential(x[0], x[1], mu, temperature), [best[0], best[1]],
                           method="Nelder-Mead", bounds=((1e-4, masses[-1]), (1e-4, waves[-1])),
                           options={"xatol": 1e-5, "fatol": 1e-11, "maxiter": 300})
            best = (float(res.x[0]), float(res.x[1]), float(res.fun))
        entry = {"mu": float(mu), "hom_mass": M_hom, "hom_potential": O_hom,
                 "mass": best[0], "wave": best[1], "potential": best[2],
                 "gap": best[2] - O_hom, "refined": bool(refined), "status": "homogeneous"}
        if confirm_below <= entry["gap"] < 0:
            # Below the spline accuracy (~1e-7 at weak field, small M): not a finding.
            entry["status"] = "marginal"
        elif entry["gap"] < confirm_below:
            # Candidates are recomputed without the spline at the automatic cutoff and twice it.
            N = vacuum_cutoff(row.B, best[1])
            direct = [row.direct(best[0], best[1], mu, temperature, n) - O_hom
                      for n in (N, min(2 * N, 1024))]
            entry["direct_gaps"] = direct
            entry["status"] = ("cdw" if max(direct) < -confirm_margin else "homogeneous")
        out.append(entry)
    row.timing["per_mu_total"] = time.perf_counter() - start
    return row, out
