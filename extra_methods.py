import qutip as qt
import numpy as np
import scipy as sp
import magpy as mp

def magnus1_mp(H_coeff, t0, tf):
    h = tf - t0
    int1 = h * H_coeff[0](t0 + h/2)
    int2 = h * H_coeff[1](t0 + h/2)
    int3 = h * H_coeff[2]
    return mp.liouvillian(int1*mp.sigmax + int2*mp.sigmay + int3*mp.sigmaz)

def mag2_mp_1(H_coeff, rho0, tlist):
    states = [mp.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        omega = magnus1_mp(H_coeff, tlist[i], tlist[i+1]) + mp._magnus_second_term([H_coeff], np.zeros((2,2)), tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = mp.unvec(states[i])
        
    states[-1] = mp.unvec(states[-1])
    
    return states

def magnus1_glq(H_coeff, t0, tf, ord):
    int1 = sp.integrate.fixed_quad(H_coeff[0], t0, tf, n=ord)[0]
    int2 = sp.integrate.fixed_quad(H_coeff[1], t0, tf, n=ord)[0]
    int3 = (tf - t0) * H_coeff[2]
    return mp.liouvillian(int1*mp.sigmax + int2*mp.sigmay + int3*mp.sigmaz)

def magnus_second_term_one_particle(H_coeff, t0, tf):
    f = H_coeff[0]
    g = H_coeff[1]

    def x(x): return x
    def q1(y, x): return g(y) - g(x)
    def q2(y, x): return f(x) - f(y)
    def q3(y, x): return f(y)*g(x) - g(y)*f(x)

    int1 = sp.integrate.dblquad(q1, t0, tf, t0, x)[0]
    int2 = sp.integrate.dblquad(q2, t0, tf, t0, x)[0]
    int3 = sp.integrate.dblquad(q3, t0, tf, t0, x)[0]

    foo = 2j*H_coeff[2]*int1*mp.sigmax + 2j*H_coeff[2]*int2*mp.sigmay + 2j*int3*mp.sigmaz
    return 0.5j * mp.liouvillian(foo)

def mag2_glqn_1(H_coeff, rho0, tlist, ord):
    states = [mp.vec(rho0)]

    for i in range(len(tlist) - 1):
        omega = magnus1_glq(H_coeff, tlist[i], tlist[i+1], ord) + magnus_second_term_one_particle(H_coeff, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(omega) @ states[i])
        states[i] = mp.unvec(states[i])

    states[-1] = mp.unvec(states[-1])

    return states