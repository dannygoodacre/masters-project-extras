from misc import *

def pade_lvn(H_coeff, rho0, tlist):
    """Evolve density matrix under Hamiltonian over increments of time using Padé approximants and scaling and squaring.

    Parameters
    ----------
    H_coeff : list/array
        Three coefficients of Pauli matrices x,y,z in Hamiltonian. First two are functions, third is a constant.
    rho0 : qutip.Qobj/ndarray
        Initial density density matrix. 
    tlist : list/array
        List of times over which to evolve density matrix.

    Returns
    -------
    list
        density matrix evaluated at each time in tlist.
    """
    H, states, time_step = setup_lvn(H_coeff, rho0, tlist)
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        states.append(sp.linalg.expm(time_step * A) @ states[i])
        states[i] = unvec(states[i])

    states[-1] = unvec(states[-1])

    return states

def krylov_lvn(H_coeff, rho0, tlist):
    """Evolve density matrix under Hamiltonian using the Krylov subspace method.

    Parameters
    ----------
    H_coeff : list/array
        Three coefficients of Pauli matrices x,y,z in Hamiltonian. First two are functions, third is a constant.
    rho0 : qutip.Qobj/ndarray
        Initial density density matrix. 
    tlist : list/array
        List of times over which to evolve density matrix.

    Returns
    -------
    list
        density matrix evaluated at each time in tlist.
    """
    H, states, time_step = setup_lvn(H_coeff, rho0, tlist) 
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        states.append(krylov_expm(time_step * A, states[i]))
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states

def magnus_lvn(H_coeff, rho0, tlist):
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        Ht = sp.integrate.quad(H_coeff[0], tlist[i], tlist[i + 1])[0]*qt.sigmax() + sp.integrate.quad(H_coeff[1], tlist[i], tlist[i + 1])[0]*qt.sigmay() + H_coeff[2]*(tlist[i + 1] - tlist[i])*qt.sigmaz()
        A = np.asarray(qt.liouvillian(Ht))
        states.append(sp.linalg.expm(A) @ states[i])
        states[i] = unvec(states[i])
    
    states[-1] = unvec(states[-1])
    
    return states

# TODO:
# Use built-in integral and commutator functions at first for Magnus
# DO SOME WRITING