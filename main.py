from methods import *

A = np.array([[1, -1j],[1j,1]])
b = np.array([2,1])
e1 = np.array([1, 0]).T

#print(V @ matrix_exp_padé(H, 5, 5) @ e1)

V, H = arnoldi(A, b)

with np.printoptions(precision = 3, suppress = True):
    print(V)
    print("V: " + str(V.shape))
    print(H)
    print("H: " + str(H.shape))