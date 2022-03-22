from methods import *
from misc import *
from extras import *
from systems.ac3 import *
import hamiltonian as ham

# Hermitian matrices and vectors for testing purposes
A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1])

states_m = magnus_lvn(H, rho0, timesteps(0, 20, 0.5**10))
values_m = traceInnerProduct(states_m, many_kron([qt.sigmax(), np.eye(2), np.eye(2)])) / 2
plt.plot(values_m)

#states_e = expm_lvn(ham.one_particle(H_coeff), rho0, timesteps(0, 20, 0.5**10))
#values_e = traceInnerProduct(states_e, qt.sigmax()) / 2
#plt.plot(values_e)

plt.show()