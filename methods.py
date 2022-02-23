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
    H, density_matrices, time_step = setup_lvn(H_coeff, rho0, tlist)
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        density_matrices.append(sp.linalg.expm(time_step * A) @ density_matrices[i])
        density_matrices[i] = unvec(density_matrices[i])

    density_matrices[-1] = unvec(density_matrices[-1])

    return density_matrices

def krylov_lvn(H_coeff, rho0, tlist):
    """Evolve density matrix under Hamiltonian over increments of time using the Krylov subspace method.

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
    H, density_matrices, time_step = setup_lvn(H_coeff, rho0, tlist) 
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        density_matrices.append(krylov_expm(time_step * A, density_matrices[i]))
        density_matrices[i] = unvec(density_matrices[i])
        
    density_matrices[-1] = unvec(density_matrices[-1])
    
    return density_matrices

def magnus_lvn(H_coeff, rho0, tlist):
    H, density_matrices, time_step = setup_lvn(H_coeff, rho0, tlist)
    
    for i in range(len(tlist) - 1):
        Ht = sp.integrate.quad(H_coeff[0], 0, tlist[i])[0]*qt.sigmax() + sp.integrate.quad(H_coeff[1], 0, tlist[i])[0]*qt.sigmay() + H_coeff[2]*tlist[i]*qt.sigmaz()
        A = np.asarray(qt.liouvillian(Ht))
        density_matrices.append(sp.linalg.expm(A) @ density_matrices[0])
        density_matrices[i+1] = unvec(density_matrices[i+1])
        
    density_matrices[0] = rho0
    
    return density_matrices

# TODO:
# Use built-in integral and commutator functions at first for Magnus
# First get it to work with only the first term of the integral
# DO SOME WRITING