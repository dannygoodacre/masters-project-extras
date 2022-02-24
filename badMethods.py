from misc import *

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
    A = np.asarray(qt.liouvillian(H))
    states = [vec(rho0)]

    fe = np.eye(A.shape[0]) + h*A

    for i in range(len(times) - 1):
        states.append(fe @ states[i])
        states[i] = unvec(states[i])

    states[-1] = unvec(states[-1])

    return states

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
    times = np.linspace(0, final_time, int(final_time/h) + 1)
    A = np.asarray(qt.liouvillian(H))
    states = [vec(rho0)]

    be = sp.linalg.inv(np.eye(A.shape[0]) - h*A)

    for i in range(len(times) - 1):
        states.append(be @ states[i])
        states[i] = unvec(states[i])

    states[-1] = unvec(states[-1])
    
    return states

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
    times = np.linspace(0, final_time, int(final_time / h) + 1)
    A = np.asarray(qt.liouvillian(H))
    states = [vec(rho0)]

    I = np.eye(A.shape[0])
    tr = sp.linalg.inv(I - (h/2)*A) @ (I + (h/2)*A)

    for i in range(len(times) - 1):
        states.append(tr @ states[i])
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])

    return states

def rk4_lvn(H_coeff, rho0, tlist):
    H, states, time_step = setup_lvn(H_coeff, rho0, tlist)
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i])))
        k1 = A @ states[i]
        k2 = A @ (states[i] + 0.5*time_step*k1)
        k3 = A @ (states[i] + 0.5*time_step*k2)
        k4 = A @ (states[i] + time_step*k3)
        
        states.append(states[i] + (1/6)*time_step*(k1 + 2*k2 + 2*k3 + k4))
        states[i] = unvec(states[i])
    
    states[-1] = unvec(states[-1])
    
    return states