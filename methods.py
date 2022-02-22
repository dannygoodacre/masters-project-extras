from misc import *

# wip
def integral(f, a, b):
    return sp.integrate.quad(f, a, b)[0]

def pade_lvn(H, rho0, tlist, args={}):
    density_matrices = [vec(rho0)] # initial condition
    h = tlist[1] - tlist[0] # time step
    
    for i in range(len(tlist) - 1):
        A = -1j * liouvillian(hamiltonian(H, tlist[i], args))
        density_matrices.append(sp.linalg.expm(h * A) @ density_matrices[i])
        density_matrices[i] = unvec(density_matrices[i])
        
    density_matrices[-1] = unvec(density_matrices[-1])
    
    return density_matrices

def krylov_lvn(H, rho0, tlist, args={}):
    density_matrices = [vec(rho0)] # initial condition
    h = tlist[1] - tlist[0] # time step
    
    for i in range(len(tlist) - 1):
        A = -1j * liouvillian(hamiltonian(H, tlist[i], args))
        density_matrices.append(krylov_expm(h * A, density_matrices[i]))
        density_matrices[i] = unvec(density_matrices[i])
        
    density_matrices[-1] = unvec(density_matrices[-1])
    
    return density_matrices

def magnus_lvn(H, rho0, tlist, args={}):
    return 0

# TODO:
# Build Magnus based method for time-dependent Hamiltonians (remember QuTiP format)
# Use built-in integral and commutator functions
# First get it to work with only the first term of the integral
# DO SOME WRITING