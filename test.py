import qutip as qt
import numpy as np
import scipy as sp
import misc
import integrate

import systems.b.a1 as a1


def expm_one_spin(data, rho0, tlist):
    h = tlist[1] - tlist[0]
    states = [misc.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(data[i][0]*qt.sigmax() + data[i][1]*qt.sigmay() + data[i][2]*qt.sigmaz()))
        states.append(sp.linalg.expm(A) @ states[i])
        states[i] = misc.unvec(states[i])
    states[-1] = misc.unvec(states[-1])
    
    return states

states = []
for k in range(1,10):
    tlist = misc.timesteps(0, 20, 0.5**k)
    data = integrate.pre_integrate(a1.H_coeff, tlist, "ip")
    states.append(expm_one_spin(data, a1.rho0, tlist))
    print(k)
np.save("data//expm_one_spin//b_a1_ip_k=1,10", states)

states = []
for k in range(1,10):
    tlist = misc.timesteps(0, 20, 0.5**k)
    data = integrate.pre_integrate(a1.H_coeff, tlist, "mp")
    states.append(expm_one_spin(data, a1.rho0, tlist))
    print(k)
np.save("data//expm_one_spin//b_a1_mp_k=1,10", states)