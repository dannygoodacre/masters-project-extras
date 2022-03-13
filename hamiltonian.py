import numpy as np
import qutip as qt

def one_particle(H_coeff):
    def H(t, args=None): return H_coeff[0](t)*qt.sigmax() + H_coeff[1](t)*qt.sigmay() + H_coeff[2]*qt.sigmaz()
    return H

def two_particle(H1_coeff, H2_coeff, J12=0):
    f1 = H1_coeff[0]
    g1 = H1_coeff[1]
    o1 = H1_coeff[2]
    
    f2 = H2_coeff[0]
    g2 = H2_coeff[1]
    o2 = H2_coeff[2]
    
    def H1(t): return qt.Qobj(f1(t)*np.kron(qt.sigmax(), np.eye(2)) + g1(t)*np.kron(qt.sigmay(), np.eye(2)) + o1*np.kron(qt.sigmaz(), np.eye(2)))
    def H2(t): return qt.Qobj(f2(t)*np.kron(np.eye(2), qt.sigmax()) + g2(t)*np.kron(np.eye(2), qt.sigmay()) + o2*np.kron(np.eye(2), qt.sigmaz()))
    H_in = 2 * np.pi * J12 * qt.Qobj(np.kron(qt.sigmax(), qt.sigmax()) + np.kron(qt.sigmay(), qt.sigmay()) + np.kron(qt.sigmaz(), qt.sigmaz()))
    
    def H(t, args=None): return H1(t) + H2(t) + H_in
    return H

def n_particle():
    # TODO
    return 0