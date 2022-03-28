import numpy as np
import qutip as qt
from misc import *

def one_particle(H_coeff):
    def H(t, args=None): return H_coeff[0](t)*qt.sigmax() + H_coeff[1](t)*qt.sigmay() + H_coeff[2]*qt.sigmaz()
    return H

def two_particle(H1_coeff, H2_coeff, HJ=0):
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

def n_particle(H_coeffs, HJ=0):
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