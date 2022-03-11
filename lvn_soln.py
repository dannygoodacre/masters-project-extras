import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import methods
import misc

# Time dependent example (the args parameter keeps the QuTiP solvers happy)
def e(t, args=None): return 10*np.exp(-(t - 10)**8 / 10**7)
def w(t, args=None): return np.exp(1j*1*(t - 10)**2)
def f(t, args=None): return e(t) * np.real(w(t))
def g(t, args=None): return e(t) * np.imag(w(t))
omega = 0.5
H_coeff = [f, g, omega]
rho0 = qt.sigmax()

# QuTiP approximation
output_qutip = qt.mesolve([omega*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]], rho0, misc.timesteps(0, 20, 0.5**10, True)).states
values_qutip = misc.traceInnerProduct(output_qutip, qt.sigmax()) / 2

tlist = misc.timesteps(0, 20, 0.5**8, True)

output = methods.krylov_lvn(H_coeff, rho0, tlist, 4)
values = misc.traceInnerProduct(output, qt.sigmax()) / 2

plt.plot(tlist, values)
plt.plot(misc.timesteps(0, 20, 0.5**10, True), values_qutip)
plt.legend(['Krylov', 'QuTiP'])
plt.show()