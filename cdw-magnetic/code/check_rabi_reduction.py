"""Exact checks for notes/07-derivation-operators.md (SymPy required).

Run from the repository root:
    python3 cdw-magnetic/code/check_rabi_reduction.py

No oscillator truncation is used. H squared is checked as a differential
operator acting on four arbitrary functions, so the magnetic commutator
is tested without a spurious finite-matrix boundary term.
"""

import math

import sympy as sp


I = sp.I
eye2, eye4 = sp.eye(2), sp.eye(4)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -I], [I, 0]])
sz = sp.diag(1, -1)
kron = sp.kronecker_product
checks = []


def check(label, expression):
    entries = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
    assert all(sp.simplify(entry) == 0 for entry in entries), label
    checks.append(label)


def comm(a, b):
    return a * b - b * a


def anti(a, b):
    return a * b + b * a


gamma = [kron(sz, sz), kron(sz, I * sx), kron(sz, I * sy)]
gamma5 = kron(sy, eye2)
beta = gamma[0]
alpha1, alpha2 = [beta * g for g in gamma[1:]]
Gamma = beta * gamma5 * gamma[1]
A, C = Gamma * alpha1, Gamma * beta
S = -I * alpha1 * alpha2
J = A * S

for mu in range(3):
    for nu in range(3):
        metric = (1, -1, -1)[mu] if mu == nu else 0
        check(f"Clifford {mu},{nu}", anti(gamma[mu], gamma[nu]) - 2 * metric * eye4)
    check(f"gamma5 anticommutes with gamma{mu}", anti(gamma5, gamma[mu]))

for label, matrix in {"alpha1": alpha1, "alpha2": alpha2, "beta": beta,
                      "Gamma": Gamma, "A": A, "C": C, "S": S, "J": J}.items():
    check(f"{label} Hermitian", matrix.H - matrix)
    check(f"{label} squared", matrix * matrix - eye4)

check("A,S commute", comm(A, S))
check("A,C anticommute", anti(A, C))
check("S,C anticommute", anti(S, C))
check("J identity", J + I * Gamma * alpha2)
for label, matrix in {"alpha1": alpha1, "alpha2": alpha2,
                      "beta": beta, "Gamma": Gamma}.items():
    check(f"J anticommutes with {label}", anti(J, matrix))

x, b, M, ky = sp.symbols("x b M k_y", real=True)
B = sp.symbols("Bcal", positive=True)
U = sp.cos(b * x) * eye4 + I * sp.sin(b * x) * gamma5
phase = sp.cos(2 * b * x) * eye4 + I * sp.sin(2 * b * x) * gamma5
check("chiral U unitary", U * U.H - eye4)
check("rotated mass", U * beta * phase * U.H - beta)
check("rotated kinetic derivative", -I * U * alpha1 * U.H.diff(x) - b * Gamma)
check("U commutes with alpha1", comm(U, alpha1))
check("U commutes with alpha2", comm(U, alpha2))

functions = sp.Matrix([sp.Function(f"f{k}")(x) for k in range(4)])


def act_h(vector):
    return (-I * alpha1 * vector.diff(x) + alpha2 * (ky + B * x) * vector
            + (M * beta + b * Gamma) * vector)


predicted_square = (-functions.diff(x, 2)
                    + ((ky + B * x) ** 2 + M ** 2 + b ** 2) * functions
                    + B * S * functions - 2 * I * b * A * functions.diff(x)
                    + 2 * b * M * C * functions)
check("H squared on arbitrary functions", act_h(act_h(functions)) - predicted_square)

hadamard = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
for j, seed_index in ((1, 0), (-1, 1)):
    projector = (eye4 + j * J) / 2
    check(f"projector {j} squared", projector ** 2 - projector)
    check(f"projector {j} rank", projector.trace() - 2)
    v = (eye4 + A) * projector * eye4[:, seed_index]
    v /= sp.sqrt((v.H * v)[0])
    W = sp.Matrix.hstack(v, C * v) * hadamard
    check(f"basis {j} orthonormal", W.H * W - eye2)
    check(f"basis {j} spans projector", W * W.H - projector)
    check(f"basis {j} A -> sx", W.H * A * W - sx)
    check(f"basis {j} C -> sz", W.H * C * W - sz)
    check(f"basis {j} S -> j sx", W.H * S * W - j * sx)
    check(f"basis {j} J -> j I", W.H * J * W - j * eye2)
    print(f"W({j:+d}) = {W}")

Q = kron(sz, eye2)
for label, matrix in {"alpha1": alpha1, "alpha2": alpha2, "Gamma": Gamma}.items():
    check(f"M sign symmetry {label}", gamma5 * matrix * gamma5 - matrix)
check("M sign symmetry beta", gamma5 * beta * gamma5 + beta)
for label, matrix in {"alpha1": alpha1, "alpha2": alpha2, "beta": beta}.items():
    check(f"b sign symmetry {label}", Q * matrix * Q - matrix)
check("b sign symmetry Gamma", Q * Gamma * Q + Gamma)

X, number = sp.symbols("X number", real=True)
g, Delta, epsilon = sp.symbols("g Delta epsilon", real=True)
h_plus = number * eye2 + (g * X + epsilon) * sx + Delta * sz
h_minus = number * eye2 + (g * X - epsilon) * sx + Delta * sz
check("AQRM sectors equivalent under oscillator parity and sz",
      sz * h_plus.subs(X, -X) * sz - h_minus)
n, spin, j = sp.symbols("n spin j", integer=True)
g_r = b / sp.sqrt(2 * B)
c = M ** 2 + b ** 2 + B
check("M=0 displacement cancels b", (c + 2 * B * (n - g_r ** 2 + j * spin / 2)).subs(M, 0)
      - (2 * B * n + B * (1 + j * spin)))

px, py, energy = sp.symbols("p_x p_y energy", real=True)
h_free = alpha1 * px + alpha2 * py + beta * M + b * Gamma
common = px ** 2 + py ** 2 + M ** 2 + b ** 2
check("B=0 characteristic polynomial",
      (energy * eye4 - h_free).det()
      - (energy ** 4 - 2 * common * energy ** 2
         + common ** 2 - 4 * b ** 2 * (px ** 2 + M ** 2)))

E, mu = sp.symbols("E mu", real=True)
T = sp.symbols("T", positive=True)
F = T * sp.log(2 * (sp.cosh(E / T) + sp.cosh(mu / T)))
check("thermal F derivative", sp.diff(F, E)
      - sp.sinh(E / T) / (sp.cosh(E / T) + sp.cosh(mu / T)))
check("thermal curvature at E=0", sp.diff(F, E, 2).subs(E, 0)
      - 1 / (T * (1 + sp.cosh(mu / T))))

print(f"PASS: {len(checks)} exact symbolic checks; SymPy {sp.__version__}.")

# Independent check of the Matsubara identity used to integrate dF/dE.
# Richardson extrapolation cancels the leading 1/cutoff tail; it does not
# truncate the Landau spectrum or approximate the Hamiltonian reduction.
def matsubara_derivative(energy_value, chemical, temperature, cutoff):
    terms = []
    for index in range(-cutoff, cutoff):
        frequency = (2 * index + 1) * math.pi * temperature
        terms.append(1 / ((frequency - 1j * chemical) ** 2 + energy_value ** 2))
    return 2 * energy_value * temperature * complex(
        math.fsum(z.real for z in terms), math.fsum(z.imag for z in terms))


for energy_value, chemical, temperature in ((0.7, 0.2, 0.4),
                                           (0.7, 1.1, 0.15),
                                           (1.3, 1.3, 0.25)):
    def fermi(argument):
        return 1 / (1 + math.exp(argument / temperature))

    target = 1 - fermi(energy_value - chemical) - fermi(energy_value + chemical)
    estimated = (2 * matsubara_derivative(energy_value, chemical, temperature, 8192)
                 - matsubara_derivative(energy_value, chemical, temperature, 4096))
    error = abs(estimated - target)
    assert error < 1e-9, (energy_value, chemical, temperature, error)
    print(f"PASS Matsubara: E={energy_value}, mu={chemical}, T={temperature}, "
          f"error={error:.3g}")
