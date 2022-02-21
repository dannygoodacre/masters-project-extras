from badMethods import *
from methods import *
import matplotlib.pyplot as plt

A = np.array([[1,2],[3,4]])
h = 0.0001

H = qt.sigmax() - qt.sigmay() + 0.5*qt.sigmaz() # Hamiltonian
rho0 = qt.sigmax() # initial condition

def f(t): return 2*t
def g(t): return t**2

density_matrices = krylov_lvn(f, g, 0.5, rho0, 5/250, 5, False)

values = traceInnerProduct(density_matrices, qt.sigmax()) / 2

plt.plot(values)
# also plot qutip mesolve solution for comparison
plt.xticks([0,50,100,150,200,250],[0,1,2,3,4,5])
plt.show()