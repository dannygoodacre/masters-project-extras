from misc import *

# wip
def integral(f, a, b):
    return sp.integrate.quad(f, a, b)[0]

def reform_lvn_equation(H, rho0, h, final_time, midpoint_time):
    times = np.linspace(0, final_time, int(final_time / h) + 1)

    if (midpoint_time):
        times = (times[1:] + times[:-1]) / 2

    return -1j*liouvillian(H), vec(rho0), times

def pade_lvn(H, rho0, h, final_time, midpoint_time):
    A, r0, times = reform_lvn_equation(H, rho0, h, final_time, midpoint_time)
    
    density_matrices = [r0] # make this an np array to save space
    
    for i in range(len(times) - 1):
        density_matrices.append(sp.linalg.expm(h * A) @ density_matrices[i]) 

    for i in range(len(times)):
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices

def krylov_lvn(H, rho0, h, final_time):
    return 0