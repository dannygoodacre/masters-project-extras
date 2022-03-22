import qutip as qt
import numpy as np
from misc import timesteps
import matplotlib.pyplot as plt
import matplotlib as mpl
from systems.ab1 import *
from matplotlib import font_manager

mpl.rcParams.update(mpl.rcParamsDefault)

font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 12,})

solution = qt.mesolve([omega*qt.sigmaz(), [qt.sigmax(), f], [qt.sigmay(), g]], rho0, timesteps(0, 5, 0.5**10), e_ops=[qt.sigmaz()])

plt.plot(solution.expect[0] / 2)
plt.xticks(np.arange(0, 5001, 1000), [0,1,2,3,4,5])
plt.yticks(np.arange(-1, 1.1, 0.5), [-1, -0.5, 0, 0.5, 1])
plt.ylim((-1.1, 1.1))
plt.xlabel('Time')
plt.ylabel('$\sigma_z$ spin component')
plt.show()

# $\frac{\langle\sigma_z\rangle}{2}$
# b = qt.Bloch()
# b.add_vectors([-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
# b.vector_width = 2
# b.zlabel = ['$\sigma_x$', '']
# b.save('test.png', format='png')