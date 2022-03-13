from misc import *

def magnus_lvn(H_coeff, rho0, tlist): # TODO: fix this like the two below. Will involve fixing magnus firs and second term
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        om = np.asarray(qt.liouvillian(magnus_first_term(H_coeff, tlist[i], tlist[i+1]) + magnus_second_term(H_coeff, tlist[i], tlist[i+1])))
        states.append(sp.linalg.expm(om) @ states[i])
        states[i] = unvec(states[i])
    
    states[-1] = unvec(states[-1])
    
    return states

def krylov_lvn(H, rho0, tlist, m, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(krylov_expm(h * A, states[i], m))
        states[i] = unvec(states[i])
    states[-1] = unvec(states[-1])
    
    return states

def expm_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(sp.linalg.expm(h * A) @ states[i])
        states[i] = unvec(states[i])
    states[-1] = unvec(states[-1])
    
    return states