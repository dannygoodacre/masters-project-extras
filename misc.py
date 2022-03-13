import numpy as np
import scipy as sp
import qutip as qt
import matplotlib.pyplot as plt

def vec(mat):
    """
    Return a vector formed by stacking columns of matrix.

    Parameters
    ----------
    mat : ndarray
        Matrix.

    Returns
    -------
    ndarray
        Vectorised form of matrix, using column-major (Fortran) ordering.

    """
    return np.asarray(mat).flatten('F')

def unvec(vec, c=None):
    """
    Return unvectorised/re-matricised vector using column-major (Fortran) ordering.

    Parameters
    ----------
    vec : ndarray
        Vector of elements.
    c : int, optional
        Desired length of columns in matrix. Leave blank if a square matrix. The default is None.

    Returns
    -------
    ndarray
        Matrix formed from vector.

    """
    vec = np.array(vec)

    if (len(vec) % 2 != 0): # odd number of elements
        if (len(vec) == 1):
            return vec
        else:
            print("Error: odd number of elements in vector. Cannot form matrix.")
            return None
    elif (c == None):
        if (np.sqrt(len(vec)).is_integer()): # matrix is square
            c = int(np.sqrt(len(vec)))
        else: # matrix is not square
            print("Error: vector cannot form a square matrix. Please provide a column length, c.")
            return None
    elif (not (len(vec) / c).is_integer()): # c does not divide length of vec
        print("Error: value of c is invalid. Cannot split vector evenly into columns of length c")
        return None

    n = int(len(vec) / c) # number of rows

    return vec.reshape((c, n), order = 'F')

def traceInnerProduct(a, b):
    """
    Return trace inner product of two square matrices.

    Parameters
    ----------
    a : ndarray
        Either an individual or array of numpy.array, numpy.matrix, or qutip.Qobj.
    b : ndarray
        Single np.array.

    Returns
    -------
    ndarray or scalar
        The value(s) of the trace of a times b.

    """
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=complex)

    try: # a is an array
        t = []
        for x in a:
            t.append(np.trace(x.conj().T @ b))

        return np.asarray(t)
    except: # a is individual
        return np.trace(a.conj().T @ b)

def timesteps(initial, final, h):
    return np.linspace(initial, final, int(final / h) + 1)        

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

def magnus_first_term(H_coeff, t0, tf):
    return sp.integrate.quad(H_coeff[0], t0, tf)[0]*qt.sigmax() + sp.integrate.quad(H_coeff[1], t0, tf)[0]*qt.sigmay() + (H_coeff[2] * (tf - t0))*qt.sigmaz()

def magnus_second_term(H_coeff, t0, tf):
    f = H_coeff[0]
    g = H_coeff[1]
    omega = H_coeff[2]
    
    def x(x) : return x
    def q1(y, x) : return g(x) - g(y)
    def q2(y, x) : return f(y) - f(x)
    def q3(y, x) : return f(x)*g(y) - g(x)*f(y)
    
    sx = omega * sp.integrate.dblquad(q1, t0, tf, t0, x)[0]
    sy = omega * sp.integrate.dblquad(q2, t0, tf, t0, x)[0]
    sz = sp.integrate.dblquad(q3, t0, tf, t0, x)[0]
    
    return sx*qt.sigmax() + sy*qt.sigmay() + sz*qt.sigmaz()

def loglog_plot(data, ref, k_range, start_of_best_fit=None):
    data = np.load(data, allow_pickle=True)
    ref = np.load(ref, allow_pickle=True)
    
    steps, errors = []
    for k in k_range:
        steps.append(0.5**k)
        ref_points = ref[::int((len(ref) - 1) / (len(data[k-1]) - 1))] # reference points that align with data points
        errors.append(np.amax(np.linalg.norm(np.subtract(ref_points, data[k-1]), axis=(1, 2))))
        
    plt.loglog(steps, errors)
    m, c = np.polyfit(np.log10(steps[start_of_best_fit:]), np.log10(errors[start_of_best_fit:]), 1)
    if start_of_best_fit is not None: 
        plt.plot(steps[start_of_best_fit:], 10**(m*np.log10(steps[start_of_best_fit:]) + c))
    return m