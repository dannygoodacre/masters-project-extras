from re import T
from methods import *

# integrating lambda nparray element-wise
# for i in range(0,4):
#     #print(vec(test1)[i](2))
#     print(integral(vec(test)[i],0,np.pi))

# create method to use Magnus expansion for time dependent A in Y'=AY

# Hermitian matrices and vectors for testing purposes
A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1,1])
e1 = np.array([1,0,0,0])

#test = np.array([[lambda t: 0, lambda t: 2*t], [lambda t: 2*t, lambda t: 0]])

test_func= lambda t: np.sin(t)
sx = np.array(qt.sigmax())

F = lambda f, t: integrate.quad(f, 0, t)[0] # integral of f from 0 to t

# Y'(t) = A(t)Y(t), initial value: Y(0)
# A(t) = f(t)B, where B is a constant matrix, f(t) is scalar function
# soln: Y(t) = expm(omega)Y(0)
# where omega = integral of A(t) from 0 to t1 + ...  (will include rest of Magnus expansion later)
# Can only evaluate Y(t) at specific time t1

Y = lambda f, B, t, Y0 : sp.linalg.expm(F(f, t) * B) @ Y0

f = lambda t: np.sin(t)
B = np.array(qt.sigmax())
Y0 = np.array([1,2])

print(Y(f, B, np.pi, Y0))
print(sp.linalg.expm(2*B) @ Y0)