import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def sigmax():
    return np.array([[0, 1], [1, 0]])


def sigmay():
    return np.array([[0, -1j], [1j, 0]])


def sigmaz():
    return np.array([[1, 0], [0, -1]])


def vec(mat):
    """
    Return vectorised form of input using column-major (Fortran) ordering.

    Parameters
    ----------
    mat : ndarray
        Matrix.

    Returns
    -------
    ndarray
        Vector.

    """

    return np.asarray(mat).flatten('F')


def unvec(vec, c=None):
    """
    Return unvectorised vector using column-major (Fortran) ordering.

    Parameters
    ----------
    vec : ndarray
        Vector of elements.
    c : int, optional
        Desired length of columns in matrix. Infers square matrix if so.
        The default is None.

    Returns
    -------
    ndarray
        Matrix.

    """

    vec = np.array(vec)

    # odd number of elements
    if (len(vec) % 2 != 0):
        if (len(vec) == 1):
            return vec
        else:
            print("Error: odd number of elements in vector. \
                  Cannot form matrix.")
            return None
    elif c is None:
        # matrix is square
        if (np.sqrt(len(vec)).is_integer()):
            c = int(np.sqrt(len(vec)))
        else:
            print("Error: vector cannot form a square matrix. \
                  Please provide a column length, c.")
            return None
    # c does not divide length of vec
    elif (not (len(vec) / c).is_integer()):
        print("Error: value of c is invalid. \
              Cannot split vector evenly into columns of length c")
        return None

    # number of rows
    n = int(len(vec) / c)

    return vec.reshape((c, n), order='F')


def liouvillian(H):
    """
    Return Liouvillian of a Hamiltonian.

    Parameters
    ----------
    H : ndarray
        Square matrix with dimension n.

    Returns
    -------
    ndarray
        Square matrix with dimension n^2.

    """

    n = H.shape[0]

    return -1j * (np.kron(np.eye(n), H) - np.kron(H.T, np.eye(n)))


def commutator(A, B, kind="normal"):
    """
    Return commutator of kind of A and B.

    Parameters
    ----------
    A : ndarray
        Square array.
    B : ndarray
        Square array.
    kind : str, optional
        kind of commutator (normal, anti), The default is "normal".

    Returns
    -------
    ndarray
        Commutator of A and B.

    """

    if kind == "normal":
        return A@B - B@A
    elif kind == "anti":
        return A@B + B@A
    else:
        raise TypeError("Unknown commutator kind " + str(kind))


def kron(*args):
    """
    Return Kronecker product of input arguments.

    Returns
    -------
    ndarray
        Kronecker product.

    Raises
    ------
    TypeError
        No input arguments.

    """

    if not args:
        raise TypeError("Requires at least one input argument")

    # input of the form [a,b,...]
    if len(args) == 1 and isinstance(args[0], list):
        mlist = args[0]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        # single
        if len(args[0].shape) == 2:
            return args[0]
        # ndarray
        else:
            mlist = args[0]
    else:
        mlist = args

    out = mlist[0]
    for m in mlist[1:]:
        out = np.kron(out, m)

    return out


def linspace(start, stop, step, dtype=None):
    """
    Return numbers spaced by specified step over specified interval.

    Parameters
    ----------
    start : array_like
        Starting value of sequence.
    stop : array_like
        End value of sequence.
    step : array_like
        Amount by which to space points in sequence.
    dtype : dtype, optional
        The type of the output array. If dtype is not given,
        then the data type is inferred from arguments. The default is None.

    Returns
    -------
    np.ndarray
        Equally spaced numbers as specified.

    """

    return (np.linspace(start, stop, int((stop - start) / step) + 1)
            .astype(dtype))


def frobenius(a, b):
    """
    Return Frobenius/trace inner product of a and b.
    Applied element-wise if a is not single.

    Parameters
    ----------
    a : ndarray
        Square array or list/array of square arrays.
    b : ndarray
        Square array.

    Returns
    -------
    ndarray or scalar
        The value(s) of the trace of a times b.

    Examples
    --------
    >>> mp.Frobenius(np.eye(2), np.ones((2,2)))
    (2+0j)
    >>> a = [np.eye(2), np.ones((2,2))]
    >>> mp.Frobenius(a, np.ones((2,2)))
    array([2.+0.j, 4.+0.j])

    """

    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=complex)

    # a is an array
    try:
        t = []
        for x in a:
            t.append(np.trace(x.conj().T @ b))
        return np.asarray(t)

    # a is single
    except:
        return np.trace(a.conj().T @ b)


def lanczos(A, b, m=None):
    """
    Perform a specified number of iteration of the Lanczos algorithm.

    Parameters
    ----------
    A : ndarray
        Matrix
    b : ndarray
        Vector
    m : int, optional
        Number of iterations, by default None

    Returns
    -------
    ndarray, ndarray
        V, T
    """
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
    
    for j in range(1, m):
        beta[j] = np.linalg.norm(W[:, j-1])
        V[:, j] = W[:, j-1] / beta[j]
        W[:, j] = A @ V[:, j]
        alpha[j] = np.vdot(W[:, j], V[:, j])
        W[:, j] = W[:, j] - alpha[j]*V[:, j] - beta[j]*V[:, j-1]

    return V, np.diagflat(alpha) + np.diagflat(beta[1:], 1) + np.diagflat(beta[1:], -1)


def krylov_expm(A, b, m=None):
    """
    Return matrix exponential of a matrix multiplied by a vector, approximated
    using the Krylov subspace method.

    Parameters
    ----------
    A : ndarray
        Matrix
    b : ndarray
        Vector
    m : int, optional
        Number of iterations, by default None

    Returns
    -------
    ndarray
        e^A * b
    """
    V, T = lanczos(A, b, m)
    return np.linalg.norm(b) * V @ sp.linalg.expm(T) @ np.eye(1, T.shape[0])[0]


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
    """Random skew-Hermitian matrix"""
    u = np.triu(np.random.rand(n, n) + 1j*np.random.rand(n, n), 1)
    u += u.conj().T
    np.fill_diagonal(u, np.diag(np.random.rand(n, n)))
    return -1j * u


def rand_herm_neg_semi_def(n):
    """Random Hermitian negative semi-definite matrix"""
    B = np.random.rand(n, n) + 1j*np.random.rand(n, n)
    return -B @ B.conj().T


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