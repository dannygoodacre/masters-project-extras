import numpy as np
import scipy as sp
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

def loglog_plot(data, ref, data_range, plot_range=None, best_fit_range=None, plot_best_fit=False, label=None):
    steps = []
    errors = []
    data_start = data_range[0]
    if plot_range is None:
        plot_range = data_range
    for k in plot_range:
        steps.append(0.5**k)
        ref_points = ref[::int((len(ref) - 1) / (len(data[k-data_start]) - 1))] # reference points that align with data points
        errors.append(np.amax(np.linalg.norm(np.subtract(ref_points, data[k-data_start]), ord=2 ,axis=(1, 2))))
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