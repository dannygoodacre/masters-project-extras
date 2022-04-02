import magpy as mp
import numpy as np
import scipy as sp
import qutip as qt
import matplotlib.pyplot as plt

def lanczos(A, b, m=None):
    n = A.shape[0]
    if m is None:
        m = n
        
    V = np.zeros((n, m), dtype='complex')
    W = np.zeros((n, m), dtype='complex')
    alpha = np.zeros((m, 1), dtype='complex')
    beta = np.zeros((m, 1), dtype='complex')
    
    V[:, 0] = b / np.linalg.norm(b)
    W[:, 0] = A @ V[:, 0]
    alpha[0] = np.vdot(W[:, 0], V[:, 0])
    W[:, 0] = W[:, 0] - alpha[0]*V[:, 0]
    
    for j in range(2, m+1):
        beta[j - 1] = np.linalg.norm(W[:, j - 2])
        V[:, j - 1] = W[:, j - 1 - 1] / beta[j - 1]
        W[:, j - 1] = A @ V[:, j - 1]
        alpha[j - 1] = np.vdot(W[:, j - 1], V[:, j - 1])
        W[:, j - 1] = W[:, j - 1] - alpha[j - 1]*V[:, j - 1] - beta[j - 1]*V[:, j - 2]

    return V, np.diagflat(alpha) + np.diagflat(beta[1:], 1) + np.diagflat(beta[1:], -1)

def krylov_expm(A, b, m=None):
    V, T = lanczos(A, b, m)
    return np.linalg.norm(b) * V @ sp.linalg.expm(T) @ np.eye(1, T.shape[0])[0]

def loglog_plot(data, ref, data_range, plot_range=None, best_fit_range=None, plot_best_fit=False, label=None, ax=None):
    steps = []
    errors = []
    data_start = data_range[0]
    if plot_range is None:
        plot_range = data_range
    for k in plot_range:
        steps.append(0.5**k)
        ref_points = ref[::int((len(ref) - 1) / (len(data[k-data_start]) - 1))] # reference points that align with data points
        errors.append(np.amax(np.linalg.norm(np.subtract(ref_points, data[k-data_start]), ord=2 ,axis=(1, 2))))
    if ax is not None:
        ax.loglog(steps, errors, label=label)
    else:
        plt.loglog(steps, errors, label=label)
    
    if best_fit_range is not None: 
        start = best_fit_range[0]
        end = best_fit_range[-1]
        
        steps = steps[start:end+1]
        errors = errors[start:end+1]
        
        m, c = np.polyfit(np.log10(steps), np.log10(errors), 1)
        if plot_best_fit:
            plt.plot(steps, 10**(m*np.log10(steps) + c))
        return m
    
    return 0

def krylov_lvn(H, rho0, tlist, m, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [mp.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(krylov_expm(h * A, states[i], m))
        states[i] = mp.unvec(states[i])
    states[-1] = mp.unvec(states[-1])
    
    return states

def expm_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [mp.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(sp.linalg.expm(h * A) @ states[i])
        states[i] = mp.unvec(states[i])
        
        if not (i % 1000):
            print(i)
        
    states[-1] = mp.unvec(states[-1])
    
    return states

def expm_one_spin(data, rho0, tlist):
    h = tlist[1] - tlist[0]
    states = [mp.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(data[i][0]*qt.sigmax() + data[i][1]*qt.sigmay() + data[i][2]*qt.sigmaz()))
        states.append(sp.linalg.expm(A) @ states[i])
        states[i] = mp.unvec(states[i])
        if not (i % 10000): 
                print(i)
    states[-1] = mp.unvec(states[-1])
    
    return states

def forward_euler_lvn(H, rho0, tlist, midpoint=False):
    states = [mp.vec(rho0)]
    h = tlist[1] - tlist[0]

    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + 0.5*midpoint*h)))
        fe = np.eye(A.shape[0]) + h*A
        
        states.append(fe @ states[i])
        states[i] = mp.unvec(states[i])

    states[-1] = mp.unvec(states[-1])

    return states

def backward_euler_lvn(H, rho0, tlist, midpoint=False):
    states = [mp.vec(rho0)]
    h = tlist[1] - tlist[0]

    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + 0.5*midpoint*h))) # Check if backward or forward is better
        be = sp.linalg.inv(np.eye(A.shape[0]) - h*A)
        
        states.append(be @ states[i])
        states[i] = mp.unvec(states[i])

    states[-1] = mp.unvec(states[-1])
    
    return states

def trapezoidal_rule_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [mp.vec(rho0)]
    I = np.eye(rho0.full().size)

    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        tr = sp.linalg.inv(I - (h/2)*A) @ (I + (h/2)*A)
        
        states.append(tr @ states[i])
        states[i] = mp.unvec(states[i])
    states[-1] = mp.unvec(states[-1])

    return states

def rk4_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [mp.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        k1 = A @ states[i]
        k2 = A @ (states[i] + 0.5*h*k1)
        k3 = A @ (states[i] + 0.5*h*k2)
        k4 = A @ (states[i] + h*k3)
        
        states.append(states[i] + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4))
        states[i] = mp.unvec(states[i])
    states[-1] = mp.unvec(states[-1])
    
    return states

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

def pade_expm(A, p, q):
    """
    Approximation of matrix exponential of A using (p,q) Padé approximants.

    Parameters
    ----------
    A : ndarray
        Square matrix.
    p : int
        Order of numerator of approximant.
    q : int
        Order of denominator of approximant.

    Returns
    -------
    ndarray
        The Padé approximant of exp(A)
    """
    N = 0
    D = 0

    f_p = sp.special.factorial(p)
    f_q = sp.special.factorial(q)
    f_p_q = sp.special.factorial(p+q)

    for i in range(0,p+1):
        N += ((sp.special.factorial(p + q - i) * f_p) / (f_p_q * sp.special.factorial(i) * sp.special.factorial(p-i))) * np.linalg.matrix_power(A,i)
    
    for i in range(0,q+1):
        D += ((sp.special.factorial(p + q - i) * f_q) / (f_p_q * sp.special.factorial(i) * sp.special.factorial(q-i))) * np.linalg.matrix_power(-A,i)
    
    return np.dot(np.linalg.inv(D),N)
   
def rand_skew_herm(n):
    u = np.triu(np.random.rand(n, n) + 1j*np.random.rand(n, n), 1)
    u += u.conj().T
    np.fill_diagonal(u, np.diag(np.random.rand(n, n)))
    return -1j * u

def rand_herm_neg_semi_def(n):
    B = np.random.rand(n, n) + 1j*np.random.rand(n, n)
    return -B @ B.conj().T

def pre_integrate(H_coeff, tlist, method):
    data = []
    if method == "SCIPY":
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = sp.integrate.quad(H_coeff[0], tlist[i], tlist[i+1])[0]
            val[1] = sp.integrate.quad(H_coeff[1], tlist[i], tlist[i+1])[0]
            val[2] = H_coeff[2] * (tlist[i+1] - tlist[i])
            data.append(val)
            if not (i % 10000): 
                print(i)
    elif method[:3] == "GLQ":
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = sp.integrate.fixed_quad(H_coeff[0], tlist[i], tlist[i+1], n=int(method[3:]))
            val[1] = sp.integrate.fixed_quad(H_coeff[1], tlist[i], tlist[i+1], n=int(method[3:]))
            val[2] = H_coeff[2] * (tlist[i+1] - tlist[i])
            data.append(val)
            if not (i % 10000): 
                print(i)
    elif method == "IP":
        h = tlist[1] - tlist[0]
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = h * H_coeff[0](tlist[i])
            val[1] = h * H_coeff[1](tlist[i])
            val[2] = h * H_coeff[2]
            data.append(val)
            if not (i % 10000): 
                print(i)
    elif method == "MP":
        h = tlist[1] - tlist[0]
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = h * H_coeff[0](tlist[i] + h/2)
            val[1] = h * H_coeff[1](tlist[i] + h/2)
            val[2] = h * H_coeff[2]
            data.append(val)
            if not (i % 10000): 
                print(i)
    else:
        print("Error: invalid method.")
        return 0
    
    return data

def one_spin(H_coeff):
    def H(t, args=None): return H_coeff[0](t)*qt.sigmax() + H_coeff[1](t)*qt.sigmay() + H_coeff[2]*qt.sigmaz()
    return H

def two_spins(H1_coeff, H2_coeff, HJ=0):
    f1 = H1_coeff[0]
    g1 = H1_coeff[1]
    o1 = H1_coeff[2]
    
    f2 = H2_coeff[0]
    g2 = H2_coeff[1]
    o2 = H2_coeff[2]
    
    def H1(t): return f1(t)*qt.tensor(qt.sigmax(), qt.identity(2)) + g1(t)*qt.tensor(qt.sigmay(), qt.identity(2)) + o1*qt.tensor(qt.sigmaz(), qt.identity(2))
    def H2(t): return f2(t)*qt.tensor(qt.identity(2), qt.sigmax()) + g2(t)*qt.tensor(qt.identity(2), qt.sigmay()) + o2*qt.tensor(qt.identity(2), qt.sigmaz())
    def H(t, args=None): return H1(t) + H2(t) + HJ
    return H

def n_spins(H_coeffs, HJ=0):
    if type(H_coeffs[0]) != type([]): # one particle
        H_coeffs = [H_coeffs]
    
    def H_total(t, args=None):
        total = HJ
        for i in range(len(H_coeffs)):
            sx = [qt.identity(2) for i in H_coeffs]
            sy = [qt.identity(2) for i in H_coeffs]
            sz = [qt.identity(2) for i in H_coeffs]
            sx[i] = qt.sigmax()
            sy[i] = qt.sigmay()
            sz[i] = qt.sigmaz()
            total = total + H_coeffs[i][0](t)*qt.tensor(sx) + H_coeffs[i][1](t)*qt.tensor(sy) + H_coeffs[i][2]*qt.tensor(sz)
        
        return total
    
    return H_total