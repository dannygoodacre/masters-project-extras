from methods import *
from temp import *

A = np.array([[1, 2],[3,4]])
b = np.array([2,1])
e1 = np.array([1, 0]).T

#print(V @ matrix_exp_padé(H, 5, 5) @ e1)

V, H = arnoldi(A, b, 1)

with np.printoptions(precision = 3, suppress = True):
    print(np.real(V))
    print("V: " + str(V.shape))
    print(np.real(H))
    print("H: " + str(H.shape))