from misc import *

# wip
def integral(f, a, b):
    return sp.integrate.quad(f, a, b)[0]

def pade_lvn(f, g, omega, rho0, h, final_time, midpoint_time):
    H, density_matrices, times = setup_lvn(f, g, omega, rho0, h, final_time, midpoint_time)
    
    for i in range(len(times) - 1):
        A = -1j * liouvillian(H(times[i]))
        density_matrices.append(sp.linalg.expm(h * A) @ density_matrices[i])
        density_matrices[i] = unvec(density_matrices[i])
        
    density_matrices[-1] = unvec(density_matrices[-1])
    
    return density_matrices

def krylov_lvn(f, g, omega, rho0, h, final_time, midpoint_time):
    H, density_matrices, times = setup_lvn(f, g, omega, rho0, h, final_time, midpoint_time)
    
    for i in range(len(times) - 1):
        A = -1j * liouvillian(H(times[i]))
        density_matrices.append(krylov_expm(h * A, density_matrices[i]))
        density_matrices[i] = unvec(density_matrices[i])
        
    density_matrices[-1] = unvec(density_matrices[-1])
    
    return density_matrices

# TODO:
# implement time dependence for Padé and Krylov methods
# H = f(t)*sx + g(t)*sy + const*sz is only form the deal with