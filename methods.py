from misc import *

def magnus_lvn(H_coeff, rho0, tlist):
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        om = np.asarray(qt.liouvillian(magnus_first_term(H_coeff, tlist[i], tlist[i+1]) + magnus_second_term(H_coeff, tlist[i], tlist[i+1])))
        states.append(sp.linalg.expm(om) @ states[i])
        states[i] = unvec(states[i])
    
    states[-1] = unvec(states[-1])
    
    return states

def krylov_lvn(H, rho0, tlist, m):
    time_step = tlist[1] - tlist[0]
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        states.append(krylov_expm(time_step * A, states[i], m))
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states

def expm_lvn(H, rho0, tlist):
    time_step = tlist[1] - tlist[0]
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        states.append(sp.linalg.expm(time_step * A) @ states[i])
        states[i] = unvec(states[i])

    states[-1] = unvec(states[-1])

    return states