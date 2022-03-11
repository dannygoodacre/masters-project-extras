import qutip as qt
import numpy as np
import methods
import misc
import hamiltonian
import matplotlib.pyplot as plt

# Chirped pulse example
def e(t): return 10*np.exp(-(t - 10)**8 / 10**7)
def w(t): return np.exp(1j * 1 * (t - 10)**2)
def f(t, args=None): return e(t) * np.real(w(t))
def g(t, args=None): return e(t) * np.imag(w(t))
H_coeff = [f, g, 1]
rho0 = qt.sigmax()

def f1(t, args=None): return 1
def g1(t, args=None): return -1
o1 = 0.5
H1_coeff = [f1, g1, o1]
def f2(t, args=None): return 1
def g2(t, args=None): return -1
o2 = 0.5
H2_coeff = [f2, g2, o2]

rho0_2 = qt.Qobj(np.kron(qt.sigmax(), qt.sigmax()))

# times = misc.timesteps(0, 20, 0.5**4, True)
# output = methods.expm_lvn(hamiltonian.two_particle(H1_coeff, H2_coeff))
# values = misc.traceInnerProduct(output, np.kron(qt.sigmax(), qt.sigmax())) / 2

# qutip
# expm initial time
# expm midpoint time

qt_times = misc.timesteps(0, 20, 0.5**10, True)
states_qutip = qt.mesolve([H_coeff[2]*qt.sigmaz(), [qt.sigmay(), H_coeff[1]], [qt.sigmax(), H_coeff[0]]], rho0, qt_times).states
value_qutip = misc.traceInnerProduct(states_qutip[-1], qt.sigmax()) / 2

# steps = []
# errors_mp_expm = []
# errors_ip_expm = []

# for k in range(1, 11):
#     states_mp_expm = methods.expm_lvn(hamiltonian.one_particle(H_coeff), rho0, misc.timesteps(0, 20, 0.5**k, True))
#     states_ip_expm = methods.expm_lvn(hamiltonian.one_particle(H_coeff), rho0, misc.timesteps(0, 20, 0.5**k, False))
    
#     value_mp_expm = misc.traceInnerProduct(states_mp_expm[-1], qt.sigmax()) / 2
#     value_ip_expm = misc.traceInnerProduct(states_ip_expm[-1], qt.sigmax()) / 2
    
#     steps.append(0.5**k)
#     errors_mp_expm.append(np.abs(value_mp_expm - value_qutip))
#     errors_ip_expm.append(np.abs(value_ip_expm - value_qutip))

# plt.loglog(steps, errors_mp_expm)
# plt.loglog(steps, errors_ip_expm)
# plt.legend(['mp expm', 'ip expm'])
# plt.xlabel('log10(h)')
# plt.ylabel('log10(error)')
# plt.title('error in midpoint expm and initial point expm against qutip, h = 0.5^k, k = 1, ..., 10')
# plt.show()

steps = []
errors_magnus = []
errors_expm = []

for k in range(1, 11):
    states_magnus = methods.magnus_lvn(H_coeff, rho0, misc.timesteps(0, 20, 0.5**k, True))
    states_expm = methods.expm_lvn(hamiltonian.one_particle(H_coeff), rho0, misc.timesteps(0, 20, 0.5**k, True))
    
    value_magnus = misc.traceInnerProduct(states_magnus[-1], qt.sigmax())
    value_expm = misc.traceInnerProduct(states_expm[-1], qt.sigmax())
    
    steps.append(0.5**k)
    errors_magnus.append(np.abs(value_magnus - value_qutip))
    errors_expm.append(np.abs(value_expm - value_qutip))
    
    print("Round " + str(k) + " done!")
    
plt.loglog(steps, errors_magnus)
plt.loglog(steps, errors_expm)
plt.legend(['2 term Magnus', 'mp expm'])
plt.xlabel('log10(h)')
plt.ylabel('log10(error)')
plt.title('error in Magnus and midpoint expm against qutip, h = 0.5^k, k = 1, ..., 10')
plt.show()