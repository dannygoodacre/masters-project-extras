from methods import *

def f(t, args=None): return 1
def g(t, args=None): return -1
omega = 1
H_coeff = [f, g, omega]

rho0 = qt.sigmax()
times = timesteps(0, 5, 0.001, True)

# qutip approximation
output_qutip = qt.mesolve([omega*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]], rho0, times).states
values_qutip = traceInnerProduct(output_qutip, qt.sigmax()) / 2

# my approximation
output = magnus_lvn(H_coeff, rho0, times)
values = traceInnerProduct(output, qt.sigmax()) / 2

plt.plot(values_qutip)
plt.plot(values)
plt.legend(['QuTiP', 'Magnus'])
plt.show()