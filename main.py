from methods import *

A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1,1])
e1 = np.array([1,0,0,0])

V, H = lanczos(E, c) # lanczos working

# with np.printoptions(precision = 3, suppress = True):
#     print(np.real(V))
#     print("V: " + str(V.shape))
#     print(np.real(H))
#     print("H: " + str(H.shape))

# TIDY AND COMMIT TO GITHUB

print(sp.linalg.expm(E)@c)
print(matrix_exp_krylov(E,c))