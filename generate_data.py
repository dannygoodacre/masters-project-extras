import qutip as qt
import numpy as np
import scipy as sp
import misc
import integrate
import methods
import time

import systems.b.a1_tenth as a1_tenth
import systems.b.a1_half as a1_half
import systems.b.a1_double as a1_double
import systems.b.a1 as a1

states = []
times = []
for k in range(1,13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    start = time.time()
    data = integrate.pre_integrate(a1_tenth.H_coeff, tlist, "SCIPY")
    states.append(methods.expm_one_spin(data, a1_tenth.rho0, tlist))
    end = time.time() - start
    times.append(end)
    print("1." + str(k))
np.save("data//b//a1_tenth//SCIPY_k=1,12", states)
np.save("data//b//a1_tenth_times//SCIPY_k=1,12", times)

for i in [1,2,10,50]:
    states = []
    times = []
    for k in range(1,13):
        tlist = misc.timesteps(0, 20, 0.5**k)
        start = time.time()
        data = integrate.pre_integrate(a1_tenth.H_coeff, tlist, "GLQ", i)
        states.append(methods.expm_one_spin(data, a1_tenth.rho0, tlist))
        end = time.time() - start
        times.append(end)
        print("2." + str(k))
    np.save("data//b//a1_tenth//GLQ" + str(i) + "_k=1,12", states)
    np.save("data//b//a1_tenth_times//GLQ" + str(i) + "_k=1,12", times)

states = []
times = []
for k in range(1,13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    start = time.time()
    data = integrate.pre_integrate(a1_tenth.H_coeff, tlist, "IP")
    states.append(methods.expm_one_spin(data, a1_tenth.rho0, tlist))
    end = time.time() - start
    times.append(end)
    print("3." + str(k))
np.save("data//b//a1_tenth//IP_k=1,12", states)
np.save("data//b//a1_tenth_times//IP_k=1,12", times)

states = []
times = []
for k in range(1,13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    start = time.time()
    data = integrate.pre_integrate(a1_tenth.H_coeff, tlist, "MP")
    states.append(methods.expm_one_spin(data, a1_tenth.rho0, tlist))
    end = time.time() - start
    times.append(end)
    print("4." + str(k))
np.save("data//b//a1_tenth//MP_k=1,12", states)
np.save("data//b//a1_tenth_times//MP_k=1,12", times)



states = []
times = []
for k in range(1,13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    start = time.time()
    data = integrate.pre_integrate(a1.H_coeff, tlist, "SCIPY")
    states.append(methods.expm_one_spin(data, a1.rho0, tlist))
    end = time.time() - start
    times.append(end)
    print("5." + str(k))
np.save("data//b//a1//SCIPY_k=1,12", states)
np.save("data//b//a1_times//SCIPY_k=1,12", times)

for i in [1,2,10,50]:
    states = []
    times = []
    for k in range(1,13):
        tlist = misc.timesteps(0, 20, 0.5**k)
        start = time.time()
        data = integrate.pre_integrate(a1.H_coeff, tlist, "GLQ", i)
        states.append(methods.expm_one_spin(data, a1.rho0, tlist))
        end = time.time() - start
        times.append(end)
        print("6." + str(k))
    np.save("data//b//a1//GLQ" + str(i) + "_k=1,12", states)
    np.save("data//b//a1_times//GLQ" + str(i) + "_k=1,12", times)

states = []
times = []
for k in range(1,13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    start = time.time()
    data = integrate.pre_integrate(a1.H_coeff, tlist, "IP")
    states.append(methods.expm_one_spin(data, a1.rho0, tlist))
    end = time.time() - start
    times.append(end)
    print("7." + str(k))
np.save("data//b//a1//IP_k=1,12", states)
np.save("data//b//a1_times//IP_k=1,12", times)

states = []
times = []
for k in range(1,13):
    tlist = misc.timesteps(0, 20, 0.5**k)
    start = time.time()
    data = integrate.pre_integrate(a1.H_coeff, tlist, "MP")
    states.append(methods.expm_one_spin(data, a1.rho0, tlist))
    end = time.time() - start
    times.append(end)
    print("8." + str(k))
np.save("data//b//a1//MP_k=1,12", states)
np.save("data//b//a1_times//MP_k=1,12", times)



tlist = misc.timesteps(0, 20, 0.5**k)

start = time.time()
data = integrate.pre_integrate(a1_tenth.H_coeff, tlist, "MP")
states = methods.expm_one_spin(data, a1_tenth.rho0, tlist)
end = time.time() - start
np.save("data//b//a1_tenth//REF_MP_k=20", states)
np.save("data//b//a1_tenth_times//REF_MP_k=20", end)
print("9")

start = time.time()
data = integrate.pre_integrate(a1_half.H_coeff, tlist, "MP")
states = methods.expm_one_spin(data, a1_half.rho0, tlist)
end = time.time() - start
np.save("data//b//a1_half//REF_MP_k=20", states)
np.save("data//b//a1_half_times//REF_MP_k-20", end)
print("10")

start = time.time()
data = integrate.pre_integrate(a1_double.H_coeff, tlist, "MP")
states = methods.expm_one_spin(data, a1_double.rho0, tlist)
end = time.time() - start
np.save("data//b//a1_double//REF_MP_k=20", states)
np.save("data//b//a1_double_times//REF_MP_k-20", end)