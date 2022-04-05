import numpy as np
import scipy as sp
import magpy as mp
import matplotlib.pyplot as plt
import misc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

mpl.rcParams.update(mpl.rcParamsDefault)
font_dirs = ['.']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'CMU Serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 12,})

nums = np.random.uniform(-40, 0, 1001)
A = np.zeros((1001, 1001))
np.fill_diagonal(A, nums)
b = np.random.rand(1001)
b /= np.linalg.norm(b)

r = -np.min(np.real(np.linalg.eigvals(A))) / 4
t = 1
print(2 * r * t)

ref = sp.linalg.expm(t * A) @ b

def err0(m): return 10 * np.exp(-m**2 / 5*r*t)
def err1(m): return (12*r*t/m**2 + 8*np.sqrt(r*t)/m) * np.exp((-0.93*m**2) / 4*r*t)
def err2(m): return (5/(r*t) + 3*np.sqrt(np.pi / (r * t))) * np.exp((r * t)**2 / m) * np.exp(-2 * r * t) * ((np.e * r * t) / m)**m

steps = mp.timesteps(1, 40, 0.5**10)
vals = []
for k in steps[:20]:
    vals.append(err1(k))
for k in steps[20:]:
    vals.append(err2(k))
plt.plot(steps, vals)

steps = mp.timesteps(1, 40, 1, dtype=int)
errors = []
for k in steps:
    errors.append(np.linalg.norm(misc.krylov_expm(t * A, b, k) - ref))
    
plt.plot(steps, errors)
plt.ylim(10**(-10), 10)
plt.xticks([0, 10,20,30,40], ['0','10', '20','30','40'])
plt.yscale('log')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.legend(['Bound','Error'])
plt.savefig("test.pdf", format="pdf")