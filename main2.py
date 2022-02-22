from badMethods import *
from methods import *
import matplotlib.pyplot as plt

A = np.array([[1,2],[3,4]])
h = 0.0001

H = qt.sigmax() - qt.sigmay() + 0.5*qt.sigmaz() # Hamiltonian
rho0 = qt.sigmax() # initial condition

def f(t, args=None): return 2*t
def g(t, args=None): return -1

args = {'A' : 2}

H = [qt.sigmax(), [qt.sigmay(), f], [qt.sigmaz(), g]] # Standard form of QuTiP Hamiltonian

# QuTiP approximation
output = qt.mesolve(H, rho0, np.linspace(0, 5, 250)).states
values = traceInnerProduct(output, qt.sigmax())

# my approximation
# output = krylov_lvn(H, rho0, timesteps(0, 5, 5/250, False), args)
# values = traceInnerProduct(output, qt.sigmax()) / 2

plt.plot(values)
# also plot qutip mesolve solution for comparison
plt.xticks([0,50,100,150,200,250],[0,1,2,3,4,5])
plt.show()
