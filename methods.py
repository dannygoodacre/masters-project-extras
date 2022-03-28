from misc import *
import qutip as qt
import misc

def krylov_lvn(H, rho0, tlist, m, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(krylov_expm(h * A, states[i], m))
        states[i] = unvec(states[i])
    states[-1] = unvec(states[-1])
    
    return states

def expm_lvn(H, rho0, tlist, midpoint=False):
    h = tlist[1] - tlist[0]
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(H(tlist[i] + midpoint*0.5*h)))
        states.append(sp.linalg.expm(h * A) @ states[i])
        states[i] = unvec(states[i])
        
        if not (i % 1000):
            print(i)
        
    states[-1] = unvec(states[-1])
    
    return states


def magnus_lvn_2(H_coeffs, rho0, tlist, HJ=qt.Qobj(0)):
    """Master equation evolution of density matrix for given Hamiltonian.
    
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

    Returns
    -------
    numpy.ndarray
        list / array of density matrices calculated at times in tlist.
    """
    states = [vec(rho0)]
    if type(H_coeffs[0]) != type([]): # one particle
        H_coeffs = [H_coeffs]
    
    for i in range(len(tlist) - 1):
        omega = magnus_first_term(H_coeffs, HJ, tlist[i], tlist[i+1]) + magnus_second_term(H_coeffs, HJ, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = unvec(states[i])
        if not (i % 1000):
            print(i)
        
    states[-1] = unvec(states[-1])
    
    return states

def magnus_lvn_1(H_coeffs, rho0, tlist, HJ=qt.Qobj(0)):
    states = [vec(rho0)]
    if type(H_coeffs[0]) != type([]):
        H_coeffs = [H_coeffs]
        
    for i in range(len(tlist) - 1):
        omega = magnus_first_term(H_coeffs, HJ, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states

def magnus_first_term(H_coeffs, HJ, t0, tf):
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

def magnus_second_term(H_coeffs, HJ, t0, tf):
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


def magnus_first_term_one_particle(H_coeff, t0, tf):
    int1 = sp.integrate.quad(H_coeff[0], t0, tf)[0]
    int2 = sp.integrate.quad(H_coeff[1], t0, tf)[0]
    
    return qt.liouvillian(int1*qt.sigmax() + int2*qt.sigmay() + H_coeff[2]*(tf - t0)*qt.sigmaz())

def magnus_second_term_one_particle(H_coeff, t0, tf):
    f = H_coeff[0]
    g = H_coeff[1]
        
    def x(x): return x
    def q1(y, x): return g(y) - g(x)
    def q2(y, x): return f(x) - f(y)
    def q3(y, x): return f(y)*g(x) - g(y)*f(x)
    
    int1 = sp.integrate.dblquad(q1, t0, tf, t0, x)[0]
    int2 = sp.integrate.dblquad(q2, t0, tf, t0, x)[0]
    int3 = sp.integrate.dblquad(q3, t0, tf, t0, x)[0]
    
    foo = 2j*H_coeff[2]*int1*qt.sigmax() + 2j*H_coeff[2]*int2*qt.sigmay() + 2j*int3*qt.sigmaz()
    return 0.5j * qt.liouvillian(foo)

def magnus_lvn_1_one_particle(H_coeff, rho0, tlist):
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        omega = magnus_first_term_one_particle(H_coeff, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states

def magnus_lvn_2_one_particle(H_coeff, rho0, tlist):
    states = [vec(rho0)]
    
    for i in range(len(tlist) - 1):
        omega = magnus_first_term_one_particle(H_coeff, tlist[i], tlist[i+1]) + magnus_second_term_one_particle(H_coeff, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states


def expm_one_spin(data, rho0, tlist):
    h = tlist[1] - tlist[0]
    states = [misc.vec(rho0)]
    
    for i in range(len(tlist) - 1):
        A = np.asarray(qt.liouvillian(data[i][0]*qt.sigmax() + data[i][1]*qt.sigmay() + data[i][2]*qt.sigmaz()))
        states.append(sp.linalg.expm(A) @ states[i])
        states[i] = misc.unvec(states[i])
        if not (i % 10000): 
                print(i)
    states[-1] = misc.unvec(states[-1])
    
    return states