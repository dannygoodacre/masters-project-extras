from badMethods import *
from methods import *
import matplotlib.pyplot as plt

# Time dependent example (the args parameter keeps the QuTiP solvers happy)
def f(t, args=None): return 2*t
def g(t, args=None): return -t**2
omega = 0.5
H_coeff = [f, g, omega]

rho0 = qt.sigmax()

# QuTiP approximation
output_qutip = qt.mesolve([0.5*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]], rho0, np.linspace(0, 5, 251)).states
values_qutip = traceInnerProduct(output_qutip, qt.sigmax())/2

# my approximation
output = krylov_lvn(H_coeff, rho0, timesteps(0, 5, 5/250, False))
values = traceInnerProduct(output, qt.sigmax()) / 2

plt.plot(values_qutip)
plt.plot(values)
plt.legend(['QuTiP', 'Me'])
plt.xticks([0,50,100,150,200,250],[0,1,2,3,4,5])
plt.show()