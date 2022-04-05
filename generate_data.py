import numpy as np
import qutip as qt
import scipy as sp
import magpy as mp
import extra_methods as em
import matplotlib.pyplot as plt
import misc

from systems.b.a1 import *

initialp = []
midp = []
sci = []
mag2_sci = []
mag2_mp = []
mag2_q3 = []

for k in range(1,11):
    tlist = mp.timesteps(0, 20, 0.5**k)
    initialp.append(misc.expm_lvn(misc.one_spin(H_coeff), rho0, tlist, False))
    midp.append(misc.expm_lvn(misc.one_spin(H_coeff), rho0, tlist, True))
    sci.append(mp.lvn_solve(H_coeff, rho0, tlist, two_terms=False))
    print(k - 0.5)
    mag2_sci.append(mp.lvn_solve(H_coeff, rho0, tlist, two_terms=True))
    mag2_mp.append(em.mag2_mp_1(H_coeff, rho0, tlist))
    mag2_q3.append(em.mag2_glqn_1(H_coeff, rho0, tlist, 3))
    print(k)

np.save("data//b//a1//IP_k=1,10", initialp)
np.save("data//b//a1//MP_k=1,10", midp)
np.save("data//b//a1//SCIPY_k=1,10", sci)
np.save("data//b//a1//MAG2_SCI1_k=1,10", mag2_sci)
np.save("data//b//a1//MAG2_MP1_k=1,10", mag2_mp)
np.save("data//b//a1//MAG2_Q3_k=1,10", mag2_q3)