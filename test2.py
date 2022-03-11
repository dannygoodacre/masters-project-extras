import qutip as qt
import numpy as np
import methods
import misc
import hamiltonian
import matplotlib.pyplot as plt

def e(t): return 10*np.exp(-(t - 10)**8 / 10**7)
def w(t): return np.exp(1j * 1 * (t - 10)**2)

def f(t, args=None): return e(t) * np.real(w(t))
def g(t, args=None): return e(t) * np.imag(w(t))
H_coeff = [f, g, 1]

def f1(t, args=None): return 1
def g1(t, args=None): return -1
o1 = 0.5
H1_coeff = [f1, g1, o1]

def f2(t, args=None): return 1
def g2(t, args=None): return -1
o2 = 0.5
H2_coeff = [f2, g2, o2]

rho0 = qt.Qobj(np.kron(qt.sigmax(), qt.sigmax()))

# times = misc.timesteps(0, 20, 0.5**4, True)
# output = methods.expm_lvn(hamiltonian.two_particle(H1_coeff, H2_coeff))
# values = misc.traceInnerProduct(output, np.kron(qt.sigmax(), qt.sigmax())) / 2

times_qutip = misc.timesteps(0, 20, 0.5**4, True)
output_qutip_2 = qt.mesolve(hamiltonian.two_particle(H1_coeff, H2_coeff), rho0, times_qutip).states
values_qutip_2 = misc.traceInnerProduct(output_qutip_2, np.kron(qt.sigmax(), qt.sigmax())) / 4

output_qutip_1 = qt.mesolve(hamiltonian.one_particle(H1_coeff), qt.sigmax(), times_qutip).states
values_qutip_1 = misc.traceInnerProduct(output_qutip_1, qt.sigmax()) / 2
# TODO: look at solving non-interacting multi-particle systems and combining them

# plt.plot(times, values)
plt.plot(times_qutip, values_qutip_2)
# plt.legend(['expm', 'QuTiP'])
plt.show()