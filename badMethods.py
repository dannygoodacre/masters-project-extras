from misc import *

def forward_euler_exp(A, h):
    """Approximation of matrix exponential of A using forward/explicit Euler.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    h : float
        Time step for method.

    Returns
    -------
    ndarray
        e^A
    """
    M = np.eye(A.shape[0]) + h*A
    p = int(1/h)
    return np.linalg.matrix_power(M, p)

def backward_euler_exp(A, h):
    """Approximation of matrix exponential of A using backward/implicit Euler.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    h : float
        Time step for method.

    Returns
    -------
    ndarray
        e^A
    """
    M = np.linalg.inv(np.eye(A.shape[0]) - h*A)
    p = int(1/h)
    return np.linalg.matrix_power(M, p)

def trapezoidal_rule_exp(A, h):
    """Approximation of matrix exponential of A using the trapezoidal rule.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    h : float
        Time step for method. 

    Returns
    -------
    ndarray
        e^A
    """
    I = np.eye(A.shape[0])
    M = np.linalg.inv(I - 0.5*h*A) @ (I + 0.5*h*A) # check if splitting and raising to power seperately is more efficient
    p = int(1/h)
    return np.linalg.matrix_power(M, p)

def forward_euler_lvn(H, rho0, h, final_time):
    """Approximation of density matrix using Forward/Explicit Euler as it
    evolves under the specified Liouville-von Neumann equation.

    Parameters
    ----------
    H : ndarray 
        n x n, Hamiltonian.
    rho0 : ndarray
        n x n, Initial density matrix.
    h : float
        Time step.
    final_time : float
        Final time

    Returns
    -------
    ndarray
        Array of density matrices evaluated across time specified.

    """
    times = np.linspace(0, final_time, int(final_time/h) + 1)

    # reformulate problem into vectorised form:  r' = -Ar = -iLr,, r(0) = r0
    A = -1j * liouvillian(H)
    r0 = vec(rho0)

    I = np.eye(A.shape[0])
    fe = I + h*A

    density_matrices = [r0]

    for i in range(len(times) - 1): # vectorised density matrices evaluated across times
        density_matrices.append(fe @ density_matrices[i])

    for i in range(len(times)): # unvectorise density matrices
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices

def backward_euler_lvn(H, rho0, h, final_time):
    """Approximation of density matrix using Backward/Implicit Euler as it
    evolves under the specified Liouville-von Neumann equation.

    Parameters
    ----------
    H : ndarray
        n x n, Hamiltonian.
    rho0 : ndarray
        n x n, Initial density matrix.
    h : float
        Time step.
    final_time : float
        Final time.

    Returns
    -------
    ndarray
        Array of density matrices evaluated across time specified.

    """
    times = np.linspace(0, final_time, int(final_time/h))

    # reformulate problem into vectorised form:  r' = -Ar = -iLr,, r(0) = r0
    A = -1j * liouvillian(H)
    r0 = vec(rho0)

    I = np.eye(A.shape[0])
    be = sp.linalg.inv(I - h*A)

    density_matrices = [r0]

    for i in range(len(times) - 1): # vectorised density matrices evaluated across times
        density_matrices.append(be @ density_matrices[i])

    for i in range(len(times)): # unvectorise density matrices
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices

def trapezoidal_rule_lvn(H, rho0, h, final_time):
    """Approximation of density matrix using Trapezoidal Ruler as it
    evolves under the specified Liouville-von Neumann equation.

    Parameters
    ----------
    H : ndarray
        n x n, Hamiltonian.
    rho0 : ndarray
        n x n, Initial density matrix.
    h : float
        Time step.
    final_time : float
        Final time.

    Returns
    -------
    ndarray
        Array of density matrices evaluated across time specified.

    """
    times = np.linspace(0, final_time, int(final_time/h))

    # reformulate problem into vectorised form:  r' = -Ar = -iLr,, r(0) = r0
    A = -1j * liouvillian(H)
    r0 = vec(rho0)

    I = np.eye(A.shape[0])
    tr = sp.linalg.inv(I - (h/2)*A) @ (I + (h/2)*A)

    density_matrices = [r0]

    for i in range(len(times) - 1): # vectorised density matrices evaluated across times
        density_matrices.append(tr @ density_matrices[i])

    for i in range(len(times)): # unvectorise density matrices
        density_matrices[i] = unvec(density_matrices[i])

    return density_matrices