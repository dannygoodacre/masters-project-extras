from methods import *
from misc import *
from extras import *

# Hermitian matrices and vectors for testing purposes
A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1])

# Time-independent Hamiltonian
H = qt.sigmax() - qt.sigmay() + 0.5*qt.sigmaz()

# Time dependent example (the args parameter keeps the QuTiP solvers happy)
def e(t, args=None): return 10*np.exp(-(t - 10)**8 / 10**7)
def w(t, args=None): return np.exp(1j*1*(t - 10)**2)

def f(t, args=None): return e(t) * np.real(w(t))
def g(t, args=None): return e(t) * np.imag(w(t))
omega = 0.5
H_coeff = [f, g, omega]
rho0 = qt.sigmax()
times = timesteps(0, 20, 0.001, True)

# QuTiP approximation
output_qutip = qt.mesolve([omega*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]], rho0, times).states
values_qutip = traceInnerProduct(output_qutip, qt.sigmax()) / 2

# my approximation
output = pade_lvn(H_coeff, rho0, times)
values = traceInnerProduct(output, qt.sigmax()) / 2

plt.plot(values_qutip)
plt.plot(values)
plt.legend(['QuTiP', 'Padé'])
plt.show()