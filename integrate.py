import numpy as np
import scipy as sp
import qutip as qt

def integrate(f, a, b, deg):
    x, w = np.polynomial.legendre.leggauss(deg)
    val = 0
    h = (b - a) / 2
    c = (a + b) / 2
    for i in range(deg):
        val += w[i] * f(h*x[i] + c)
    val *= h
    
    return val

def pre_integrate(H_coeff, tlist, method, deg=100):
    integrals = []
    if method == "SCIPY":
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = sp.integrate.quad(H_coeff[0], tlist[i], tlist[i+1])[0]
            val[1] = sp.integrate.quad(H_coeff[1], tlist[i], tlist[i+1])[0]
            val[2] = H_coeff[2] * (tlist[i+1] - tlist[i])
            integrals.append(val)
            if not (i % 10000): 
                print(i)
    elif method == "GLQ":
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = integrate(H_coeff[0], tlist[i], tlist[i+1], deg)
            val[1] = integrate(H_coeff[1], tlist[i], tlist[i+1], deg)
            val[2] = H_coeff[2] * (tlist[i+1] - tlist[i])
            integrals.append(val)
            if not (i % 10000): 
                print(i)
    elif method == "IP":
        h = tlist[1] - tlist[0]
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = h * H_coeff[0](tlist[i])
            val[1] = h * H_coeff[1](tlist[i])
            val[2] = h * H_coeff[2]
            integrals.append(val)
            if not (i % 10000): 
                print(i)
    elif method == "MP":
        h = tlist[1] - tlist[0]
        for i in range(len(tlist) - 1):
            val = [0, 0, 0]
            val[0] = h * H_coeff[0](tlist[i] + h/2)
            val[1] = h * H_coeff[1](tlist[i] + h/2)
            val[2] = h * H_coeff[2]
            integrals.append(val)
            if not (i % 10000): 
                print(i)
    else:
        print("Error: invalid method.")
        return 0
    
    return integrals