import magpy as mp
import numpy as np
import qutip as qt

a = qt.sigmax()
b = qt.sigmay()

c = np.array([a,b])
print(np.linalg.norm(qt.tensor(a,b,a,a,a,a).full() - mp.kron(a,b,a,a,a,a)))