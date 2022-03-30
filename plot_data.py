import qutip as qt
import numpy as np
import misc
import hamiltonian
import matplotlib.pyplot as plt
import matplotlib as mpl
from systems.b.a1 import *
from matplotlib import font_manager

mpl.rcParams.update(mpl.rcParamsDefault)
font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 12,})

states12 = qt.mesolve(hamiltonian.one_particle(H_coeff), rho0, misc.timesteps(0, 5, 0.5**12), e_ops=[qt.sigmax(), qt.sigmay(), qt.sigmaz()])

xcomp = states12.expect[0] / 2
ycomp = states12.expect[1] / 2
zcomp = states12.expect[2] / 2

b = qt.Bloch()

# 1100, 1300, 10

for i in range(11250, 13000, 50):
    b.add_points([xcomp[i], ycomp[i], zcomp[i]])

b.add_vectors([xcomp[12950], ycomp[12950], zcomp[12950]])

# $\frac{\langle\sigma_z\rangle}{2}$
b.vector_width = 2
b.xlabel = ['$\sigma_x$', '']
b.ylabel = ['$\sigma_y$', '']
b.zlabel = ['$\sigma_z$', '']
b.view = [-60, 25]
b.point_color = ['b']
b.point_marker = ['o']
b.point_size = [25]
b.save('bloch.svg', format='svg')