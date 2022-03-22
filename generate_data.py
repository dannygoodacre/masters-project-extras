import numpy as np
import methods
import badMethods
import misc
import hamiltonian
from systems.ad5 import *
import systems.ae5 as ae5
import systems.aa1 as aa1

# 5 spin, non interacting
magi = []
for k in range(1, 13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    magi.append(methods.magnus_lvn(H_coeffs, rho0, tlist))
    print("Magnus 5." + str(k))
np.save("data//ad5_magi_k=1,12", magi)

tlist = misc.timesteps(0, 20, 0.5**20)
qutip = qt.mesolve(hamiltonian.n_particle(H_coeffs), rho0, tlist).states
np.save("data//ad5_qutip_k=20_ref", qutip)

expmi = []
for k in range(1, 13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    expmi.append(methods.expm_lvn(hamiltonian.n_particle(H_coeffs), rho0, tlist, False))
    print("Expm i 5." + str(k))
np.save("data//ad5_expmi_k=1,12", expmi)

expmm = []
for k in range(1, 13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    expmm.append(methods.expm_lvn(hamiltonian.n_particle(H_coeffs), rho0, tlist, True))
    print("Expm m 5." + str(k))
np.save("data//ad5_expmm_k=1,12", expmm)

# 5 spin, interacting
magi = []
for k in range(1, 13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    magi.append(methods.magnus_lvn(ae5.H_coeffs, ae5.rho0, tlist, ae5.HJ))
    print("Magnus 5." + str(k))
np.save("data//ae5_magi_k=1,12", magi)

tlist = misc.timesteps(0, 20, 0.5**20)
qutip = qt.mesolve(hamiltonian.n_particle(ae5.H_coeffs, ae5.HJ), ae5.rho0, tlist).states
np.save("data//ae5_qutip_k=20_ref", qutip)

# 1 spin
magi = []
for k in range(1, 13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    magi.append(methods.magnus_lvn(aa1.H_coeffs, aa1.rho0, tlist))
    print("Magnus 1." + str(k))
np.save("data//aa1_magi_k=1,12", magi)

