import numpy as np
import scipy as sp
import misc


def krylov_lvn(H, rho0, tlist, m, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [misc.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(misc.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(misc.krylov_expm(h * A, states[i], m))
        states[i] = misc.unvec(states[i])
    states[-1] = misc.unvec(states[-1])
    
    return states


def expm_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [misc.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(misc.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(sp.linalg.expm(h * A) @ states[i])
        states[i] = misc.unvec(states[i])
    states[-1] = misc.unvec(states[-1])
    
    return states


def forward_euler_lvn(H, rho0, tlist, midpoint=False):
    states = [misc.vec(rho0)]
    h = tlist[1] - tlist[0]

    for i in range(len(tlist) - 1):
        A = np.asarray(misc.liouvillian(H(tlist[i] + 0.5*midpoint*h)))
        fe = np.eye(A.shape[0]) + h*A
        
        states.append(fe @ states[i])
        states[i] = misc.unvec(states[i])

    states[-1] = misc.unvec(states[-1])

    return states


def backward_euler_lvn(H, rho0, tlist, midpoint=False):
    states = [misc.vec(rho0)]
    h = tlist[1] - tlist[0]

    for i in range(len(tlist) - 1):
        A = np.asarray(misc.liouvillian(H(tlist[i] + 0.5*midpoint*h)))
        be = sp.linalg.inv(np.eye(A.shape[0]) - h*A)
        
        states.append(be @ states[i])
        states[i] = misc.unvec(states[i])

    states[-1] = misc.unvec(states[-1])
    
    return states


def trapezoidal_rule_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [misc.vec(rho0)]
    I = np.eye(rho0.full().size)

    for i in range(len(tlist) - 1):
        A = np.asarray(misc.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        tr = sp.linalg.inv(I - (h/2)*A) @ (I + (h/2)*A)
        
        states.append(tr @ states[i])
        states[i] = misc.unvec(states[i])
    states[-1] = misc.unvec(states[-1])

    return states


def rk4_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [misc.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(misc.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        k1 = A @ states[i]
        k2 = A @ (states[i] + 0.5*h*k1)
        k3 = A @ (states[i] + 0.5*h*k2)
        k4 = A @ (states[i] + h*k3)
        
        states.append(states[i] + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4))
        states[i] = misc.unvec(states[i])
    states[-1] = misc.unvec(states[-1])
    
    return states