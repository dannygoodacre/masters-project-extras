import qutip as qt
import numpy as np
import scipy as sp
import misc
import integrate
import methods
import time
import hamiltonian

import matplotlib.pyplot as plt

import systems.b.a1_tenth as a1_tenth
import systems.b.a1_half as a1_half
import systems.b.a1_double as a1_double
import systems.b.a1 as a1

# states = []
# for k in range(1,11):
#     tlist = misc.timesteps(0, 20, 0.5**k)
#     states.append(methods.glq(a1.H_coeff, a1.rho0, tlist, 25))
#     print(k)
# np.save("data//b//a1//GLQ25_k=1,10.npy", states)

# ref = np.load("data//b//a1//REF_MP_k=20.npy", allow_pickle=True)
# data = np.load("data//b//a1//GLQ25_k=1,10.npy", allow_pickle=True)
# data1 = np.load("data//b//a1//MP_k=1,12.npy", allow_pickle=True)

# misc.loglog_plot(data, ref, range(1,11), label="glq25")
# misc.loglog_plot(data1, ref, range(1,13), label="mp")
# plt.legend()
# plt.show()

#Generate data
tlist = misc.timesteps(0, 20, 0.5**20)
start = time.time()
data = integrate.pre_integrate(a1_double.H_coeff, tlist, "MP")
states = methods.expm_one_spin(data, a1_double.rho0, tlist)
end = time.time() - start
np.save("data//b//a1_double//REF_MP_k=20", states)
np.save("data//b//a1_double_times//REF_MP_k-20", end)