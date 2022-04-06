import numpy as np
import scipy as sp
import qutip as qt

# TODO: Remove QuTiP dependency

def vec(mat):
    """
    Stacks columns of matrix into vector using column-major (Fortran) ordering.

    Parameters
    ----------
    mat : ndarray
        Matrix.

    Returns
    -------
    ndarray
        Vectorised matrix.
        
    """
    return np.asarray(mat).flatten('F')

def unvec(vec, c=None):
    """
    Unvectorised vector using column-major (Fortran) ordering.

    Parameters
    ----------
    vec : ndarray
        Vector of elements.
    c : int, optional
        Desired length of columns in matrix. Infers square matrix if so. The default is None.

    Returns
    -------
    ndarray
        Matrix.
        
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

def liouvillian(H):
    """
    Liouvillian of a given Hamiltonian.

    Parameters
    ----------
    H : ndarray
        Square matrix with dimension n.

    Returns
    -------
    ndarray
        Square matrix with dimension n^2.

    """
    H = np.asarray(H)
    n = H.shape[0]

    return -1j * (np.kron(np.eye(n),H) - np.kron(H.T,np.eye(n)))

def kron(*args):
    """
    Calculates the Kronecker product of input arguments.

    Returns
    -------
    ndarray
        Kronecker product of input arguments.

    Raises
    ------
    TypeError
        No input arguments.
        
    """
    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], list):
        mlist = args[0]
    elif len(args) == 1 and isinstance(args[0], np.ndarray):
        if len(args[0].shape) == 2: # single
            return args[0]
        else: # ndarray
            mlist = args[0]
    else:
        mlist = args
    
    out = mlist[0]    
    for m in mlist[1:]:
        out = np.kron(out, m)
        
    return out

def timesteps(start, stop, step, dtype=None):
    """
    Numbers spaced by specified step over specified interval.

    Parameters
    ----------
    start : array_like
        Starting value of sequence.
    stop : array_like
        End value of sequence.
    step : array_like
        Amount by which to space points in sequence.
    dtype : dtype, optional
        The type of the output array. If dtype is not given, then the data type is inferred from arguments. The default is None.

    Returns
    -------
    np.ndarray
        Equally spaced numbers as specified.
        
    """
    return np.linspace(start, stop, int((stop - start) / step) + 1).astype(dtype)

def Frobenius(a, b):
    """
    Frobenius/trace inner product of a and b. Applied element-wise if a is not single.

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

def magnus1(H_coeffs, HJ, t0, tf):
    omega1 = (tf - t0) * HJ
    for j in range(len(H_coeffs)):
        Ijx = [qt.identity(2) for i in H_coeffs]
        Ijy = [qt.identity(2) for i in H_coeffs]
        Ijz = [qt.identity(2) for i in H_coeffs]
        Ijx[j] = qt.sigmax()
        Ijy[j] = qt.sigmay()
        Ijz[j] = qt.sigmaz()
        
        omega1 = omega1 + sp.integrate.quad(H_coeffs[j][0], t0, tf)[0]*qt.tensor(Ijx) + sp.integrate.quad(H_coeffs[j][1], t0, tf)[0]*qt.tensor(Ijy) + H_coeffs[j][2]*(tf - t0)*qt.tensor(Ijz)
    
    return qt.liouvillian(omega1)

def magnus2(H_coeffs, HJ, t0, tf):
    omega2 = 0
    for j in range(len(H_coeffs)):
        Ijx = [qt.identity(2) for i in H_coeffs]
        Ijy = [qt.identity(2) for i in H_coeffs]
        Ijz = [qt.identity(2) for i in H_coeffs]
        Ijx[j] = qt.sigmax()
        Ijy[j] = qt.sigmay()
        Ijz[j] = qt.sigmaz()
        
        c1 = 2j*H_coeffs[j][2]*qt.tensor(Ijx) + qt.commutator(qt.tensor(Ijy), HJ)
        c2 = 2j*H_coeffs[j][2]*qt.tensor(Ijy) + qt.commutator(HJ, qt.tensor(Ijx))
        c3 = 2j * qt.tensor(Ijz)
        
        f = H_coeffs[j][0]
        g = H_coeffs[j][1]
        
        def x(x): return x
        def q1(y, x): return g(y) - g(x)
        def q2(y, x): return f(y) - f(x)
        def q3(y, x): return f(y)*g(x) - g(y)*f(x)
        
        int1 = sp.integrate.dblquad(q1, t0, tf, t0, x)[0]
        int2 = sp.integrate.dblquad(q2, t0, tf, t0, x)[0]
        int3 = sp.integrate.dblquad(q3, t0, tf, t0, x)[0]
        
        omega2 = omega2 + int1*c1 - int2*c2 + int3*c3

    return 0.5j * qt.liouvillian(omega2)

def lvn_solve(H_coeffs, rho0, tlist, HJ, two_terms=True):
    """
    Liouville-von Neumann evolution of density matrix for given Hamiltonian.
    
    For n particles, the Hamiltonian takes the form: 
    sum_{k=1}^{n} Id otimes  ... otimes (f_k(t)*sigmax + g_k(t)*sigmay + omega_k*sigmaz) otimes  ... otimes Id,
    where k denotes position in the kronecker product.
    
    For one particle the Hamiltonian takes the form f(t)*sigmax + g(t)*sigmay + omega*sigmaz
        
    H_coeffs then takes the form [[f1, g1, omega1], [f2, g2, omega2], ...], or [f, g, omega] for a single particle.
    f and g must be functions and the omegas are scalar constants.
    
    Parameters
    ----------
    H_coeffs : list / array
        list of coefficients that form Hamiltonian.
    rho0 : qutip.Qobj
        Initial density matrix.
    tlist : list / array
        Times at which to calculate density matrices.
    HJ : qutip.Qobj, optional
        Interacting part of Hamiltonian, by default None.
    two_terms : bool, optional
        Whether or not to use two terms of Magnus expansion.
    Returns
    -------
    numpy.ndarray
        list / array of density matrices calculated at times in tlist.
        
    """
    states = [vec(rho0)]
    if type(H_coeffs[0]) != type([]): # one particle
        H_coeffs = [H_coeffs]
    
    for i in range(len(tlist) - 1):
        omega = magnus1(H_coeffs, HJ, tlist[i], tlist[i+1]) + two_terms*magnus2(H_coeffs, HJ, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states