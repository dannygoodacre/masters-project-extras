"""Methods which are useful but not currently used."""

from misc import *
from badMethods import *
import matplotlib.pyplot as plt

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

def liouvillian(H):
    """
    Return Liouvillian of system given the Hamiltonian.

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

    return (np.kron(np.eye(n),H) - np.kron(H.T,np.eye(n)))

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

def example_numerical_comparison():
    H = qt.sigmax() - qt.sigmay() + 0.5*qt.sigmaz() # Hamiltonian
    rho0 = qt.sigmax() # initial condition

    final_time = 5
    h = 5/250
    den_mat_actual = qt.mesolve(H, rho0, np.linspace(0, final_time, int(final_time/h))).states
    den_mat_fe = forward_euler_lvn(H, rho0, h, final_time)
    den_mat_be = backward_euler_lvn(H,rho0,h,final_time)
    den_mat_tr = trapezoidal_rule_lvn(H, rho0, h, final_time)

    actual_values = traceInnerProduct(den_mat_actual, qt.sigmax())/2
    fe_values = traceInnerProduct(den_mat_fe, qt.sigmax())/2
    be_values = traceInnerProduct(den_mat_be, qt.sigmax())/2
    tr_values = traceInnerProduct(den_mat_tr, qt.sigmax())/2

    plt.plot(actual_values)
    plt.plot(fe_values)
    plt.plot(be_values)
    plt.plot(tr_values)

    plt.title('Comparison of Numerical Methods\n for Solving the LvN Equation (h = 0.02)')
    plt.legend(['Actual solution','Forward Euler','Backward Euler','Trapezoidal rule'])
    plt.xticks([0,50,100,150,200,250],[0,1,2,3,4,5])
    plt.xlabel('Time')
    plt.ylabel('x-component of spin')
    #plt.savefig('ExampleNumericalComparison.eps', format='eps')
    plt.show()