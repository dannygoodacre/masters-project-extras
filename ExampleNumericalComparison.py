import numpy as np
import misc
import qutip as qt
import matplotlib.pyplot as plt
from badMethods import *

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