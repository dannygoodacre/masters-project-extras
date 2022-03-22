from misc import *

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
    states[-1] = unvec(states[-1])
    
    return states

def magnus_lvn(H_coeffs, rho0, tlist, HJ=None):
    states = [vec(rho0)]
    if HJ is None:
        HJ = np.zeros(rho0.shape)
    
    for i in range(len(tlist) - 1):
        omega = magnus_first_term(H_coeffs, HJ, tlist[i], tlist[i+1]) + magnus_second_term(H_coeffs, HJ, tlist[i], tlist[i+1])
        states.append(sp.linalg.expm(np.asarray(omega)) @ states[i])
        states[i] = unvec(states[i])
        
    states[-1] = unvec(states[-1])
    
    return states

def magnus_first_term(H_coeffs, HJ, t0, tf):
    omega1 = (tf - t0) * HJ
    for j in range(len(H_coeffs)):
        Ijx = [np.eye(2) for i in H_coeffs]
        Ijy = [np.eye(2) for i in H_coeffs]
        Ijz = [np.eye(2) for i in H_coeffs]
        Ijx[j] = qt.sigmax().full()
        Ijy[j] = qt.sigmay().full()
        Ijz[j] = qt.sigmaz().full()
        
        omega1 = omega1 + sp.integrate.quad(H_coeffs[j][0], t0, tf)[0]*many_kron(Ijx) + sp.integrate.quad(H_coeffs[j][1], t0, tf)[0]*many_kron(Ijy) + H_coeffs[j][2]*(tf - t0)*many_kron(Ijz)
    
    return qt.liouvillian(qt.Qobj(omega1))

def magnus_second_term(H_coeffs, HJ, t0, tf):
    omega2 = 0
    for j in range(len(H_coeffs)):
        Ijx = [np.eye(2) for i in H_coeffs]
        Ijy = [np.eye(2) for i in H_coeffs]
        Ijz = [np.eye(2) for i in H_coeffs]
        Ijx[j] = qt.sigmax().full()
        Ijy[j] = qt.sigmay().full()
        Ijz[j] = qt.sigmaz().full()
        
        c1 = 2j*H_coeffs[j][2]*many_kron(Ijx) + qt.commutator(many_kron(Ijy), HJ)
        c2 = -2j*H_coeffs[j][2]*many_kron(Ijy) + qt.commutator(HJ, many_kron(Ijx))
        c3 = 2j * many_kron(Ijz)
        
        f = H_coeffs[j][0]
        g = H_coeffs[j][1]
        
        def x(x): return x
        def q1(y, x): return g(y) - g(x)
        def q2(y, x): return f(y) - f(x)
        def q3(y, x): return f(y)*g(x) - g(y)*f(x)
        
        int1 = sp.integrate.dblquad(q1, t0, tf, t0, x)[0]
        int2 = sp.integrate.dblquad(q2, t0, tf, t0, x)[0]
        int3 = sp.integrate.dblquad(q3, t0, tf, t0, x)[0]
        
        omega2 = omega2 + int1*c1 + int2*c2 + int3*c3
        
    return 0.5j * qt.liouvillian(qt.Qobj(omega2))