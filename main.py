from methods import *
from misc import *
from extras import *
import hamiltonian as ham

import matplotlib as mpl
from systems.ag1 import *
from matplotlib import font_manager

# Hermitian matrices and vectors for testing purposes
A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1])

mpl.rcParams.update(mpl.rcParamsDefault)

font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 12,})

#plt.xticks(np.arange(0, 5001, 1000), [0,1,2,3,4,5])
# plt.yticks(np.arange(-2, 2.1, 1), [-2, -1, 0, 1, 2])
# plt.ylim((-2.2, 2.2))
# plt.xlabel('Time')
# plt.ylabel('$\sigma_z$ spin component')

# states_e = qt.mesolve(ham.one_particle(H_coeff), rho0, timesteps(0, 4, 0.5**10)).states
# values_e = traceInnerProduct(states_e, qt.sigmaz()) / 2
# plt.plot(timesteps(0, 4, 0.5**10),values_e, label='QuTiP')

# states_m = trapezoidal_rule_lvn(ham.one_particle(H_coeff), rho0, timesteps(0, 4, 0.5**4), True)
# values_m = traceInnerProduct(states_m, qt.sigmaz()) / 2
# plt.plot(timesteps(0, 4, 0.5**4),values_m, label='Trapezoidal Rule')

# plt.legend()
# plt.show()

qutip = np.load('data//aa1_qutip_k=20_ref.npy', allow_pickle=True)
print(loglog_plot(np.load('data//aa1_trapi_k=1,12.npy', allow_pickle=True)[4:], qutip, range(1,9), range(1,9), range(5,9), False, label='Initial point'))
print(loglog_plot(np.load('data//aa1_trapm_k=1,12.npy', allow_pickle=True)[4:], qutip, range(1,9), range(1,9), range(5,9), False, label='Midpoint'))
plt.xlabel('Time step')
plt.ylabel('Maximum error')
plt.legend()
plt.show()