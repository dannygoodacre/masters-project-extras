from math import sqrt
import numpy as np
import scipy.linalg as la
import qutip as qt

def vec(mat):
    """
    Return a vector formed by stacking columns of matrix.

    Parameters
    ----------
    mat : array_like
        Matrix.

    Returns
    -------
    array_like
        Vectorised form of matrix, using column-major (Fortran) ordering.

    """
    return np.asarray(mat).flatten('F')

def unvec(vec, c = None):
    """
    Return unvectorised/re-matricised vector using column-major (Fortran) ordering.

    Parameters
    ----------
    vec : array_like
        Vector of elements.
    c : int, optional
        Desired length of columns in matrix. Leave blank if a square matrix. The default is None.

    Returns
    -------
    array_like
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
        if (sqrt(len(vec)).is_integer()): # matrix is square
            c = int(sqrt(len(vec)))
        else: # matrix is not square
            print("Error: vector cannot form a square matrix. Please provide a column length, c.")
            return None
    elif (not (len(vec)/c).is_integer()): # c does not divide length of vec
        print("Error: value of c is invalid. Cannot split vector evenly into columns of length c")
        return None

    n = int(len(vec)/c) # number of rows

    return vec.reshape((c,n),order='F')

def liouvillian(H):
    """
    Return Liouvillian of system given the Hamiltonian.

    Parameters
    ----------
    H : array_like
        Square matrix with dimension n.

    Returns
    -------
    array_like
        Square matrix with dimension n^2.

    """
    H = np.asarray(H)
    n = int(sqrt(H.size)) # dimension of H

    return (np.kron(np.eye(n),H) - np.kron(H.T,np.eye(n)))

def traceInnerProduct(a, b):
    """
    Return trace inner product of two square matrices.

    Parameters
    ----------
    a : array_like
        Either an individual or array of numpy.array, numpy.matrix, or qutip.Qobj.
    b : array_like
        Single np.array.

    Returns
    -------
    ndarray or scalar
        The value(s) of the trace of a times b.

    """
    a = np.asarray(a)
    b = np.asarray(b)

    try: # a is an array
        t = []
        for x in a:
            t.append(np.trace(x@b))

        return np.asarray(t)
    except: # a is individual
        return np.trace(a@b)

def BlochSphereCoordinates(H, rho0, tlist):
    """
    Return 3D coordinates for trace inner product of density matrix in each spin direction (x,y,z) at times tlist.
    Coordinates are normalised so they lie on the surface of a Bloch sphere.

    Parameters
    ----------
    H : qutip.Qobj
        System Hamiltonian.
    rho0 : qutip.Qobj
        Initial density matrix.
    tlist : list/array
        List of times for t.

    Returns
    -------
    int numpy.array, int numpy.array, int numpy.array
        3D coordinates as described above.

    """
    density_matrices = qt.mesolve(H, rho0, tlist)

    x = np.real(traceInnerProduct(density_matrices.states, qt.sigmax()))/2
    y = np.real(traceInnerProduct(density_matrices.states, qt.sigmay()))/2
    z = np.real(traceInnerProduct(density_matrices.states, qt.sigmaz()))/2

    return x,y,z