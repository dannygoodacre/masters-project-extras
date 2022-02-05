from ylp import *

def forward_euler_lvn(H, rho0, h, final_time):
    """Approximation of density matrix using Forward/Explicit Euler as it
    evolves under the specified Liouville-von Neumann equation

    Parameters
    ----------
    H : array_like
        Hamiltonian.
    rho0 : array_like
        Initial density matrix.
    h : int
        Time step.
    final_time : int
        Final time

    Returns
    -------
    density_matrices : numpy.array
        Array of density matrices evaluated across time specified.

    """
    times = np.linspace(0, final_time, int(final_time/h))

    # reformulate problem into vectorised form:  r' = -Ar = -iLr,, r(0) = r0
    A = -1j * liouvillian(H)
    r0 = vec(rho0)

    I = np.eye(int(sqrt(A.size)))
    fe = I + h*A

    density_matrices = [r0]

    for i in range(len(times) - 1): # vectorised density matrices evaluated across times
        density_matrices.append(fe @ density_matrices[i])

    for i in range(len(times)): # unvectorise density matrices
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices

def backward_euler_lvn(H, rho0, h, final_time):
    """Approximation of density matrix using Backward/Implicit Euler as it
    evolves under the specified Liouville-von Neumann equation

    Parameters
    ----------
    H : array_like
        Hamiltonian.
    rho0 : array_like
        Initial density matrix.
    h : int
        Time step.
    final_time : int
        Final time

    Returns
    -------
    density_matrices : numpy.array
        Array of density matrices evaluated across time specified.

    """
    times = np.linspace(0, final_time, int(final_time/h))

    # reformulate problem into vectorised form:  r' = -Ar = -iLr,, r(0) = r0
    A = -1j * liouvillian(H)
    r0 = vec(rho0)

    I = np.eye(int(sqrt(A.size)))
    be = la.inv(I - h*A)

    density_matrices = [r0]

    for i in range(len(times) - 1): # vectorised density matrices evaluated across times
        density_matrices.append(be @ density_matrices[i])

    for i in range(len(times)): # unvectorise density matrices
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices

def trapezoidal_rule_lvn(H, rho0, h, final_time):
    """Approximation of density matrix using Trapezoidal Ruler as it
    evolves under the specified Liouville-von Neumann equation

    Parameters
    ----------
    H : array_like
        Hamiltonian.
    rho0 : array_like
        Initial density matrix.
    h : int
        Time step.
    final_time : int
        Final time

    Returns
    -------
    density_matrices : numpy.array
        Array of density matrices evaluated across time specified.

    """
    times = np.linspace(0, final_time, int(final_time/h))

    # reformulate problem into vectorised form:  r' = -Ar = -iLr,, r(0) = r0
    A = -1j * liouvillian(H)
    r0 = vec(rho0)

    I = np.eye(int(sqrt(A.size)))
    tr = la.inv(I - (h/2)*A) @ (I + (h/2)*A)

    density_matrices = [r0]

    for i in range(len(times) - 1): # vectorised density matrices evaluated across times
        density_matrices.append(tr @ density_matrices[i])

    for i in range(len(times)): # unvectorise density matrices
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices