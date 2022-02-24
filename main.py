from methods import *
from misc import *
from extras import *

# Hermitian matrices and vectors for testing purposes
A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1,1])
e1 = np.array([1,0,0,0])

# Time dependent example (the args parameter keeps the QuTiP solvers happy)
def f(t, args=None): return 2*t
def g(t, args=None): return -t**2
omega = 0.5
H_coeff = [f, g, omega]

rho0 = qt.sigmax()

print(qt.commutator(A, A))

# QuTiP approximation
# output_qutip = qt.mesolve([omega*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]], rho0, timesteps(0, 5, 1/250, True)).states
# values_qutip = traceInnerProduct(output_qutip, qt.sigmax()) / 2

# my approximation
# output = pade_lvn(H_coeff, rho0, timesteps(0, 5, 1/250, True))
# values = traceInnerProduct(output, qt.sigmax()) / 2

# plt.plot(values_qutip)
# plt.plot(values)
# plt.legend(['QuTiP', 'Me'])
# plt.xticks([0,50,100,150,200,250],[0,1,2,3,4,5])
# plt.show()