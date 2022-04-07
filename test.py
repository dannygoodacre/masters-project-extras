import magpy as mp
import numpy as np
import qutip as qt
import systems.b.a1 as a1
import matplotlib.pyplot as plt

tlist = mp.linspace(0, 20, 0.5**5)
states = mp.lvn_solve(a1.H_coeff, a1.rho0, tlist)
values = mp.Frobenius(states, a1.rho0)

plt.plot(tlist, values)
plt.show()