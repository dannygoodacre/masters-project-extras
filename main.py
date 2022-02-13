from methods import *

# Hermitian matrices and vectors for testing purposes
A = np.array([[-1, 1-2j, 0], [1+2j, 0, -1j], [0, 1j, 1]])
B = np.array([[2, -1j], [1j, 1]])
C = np.array([[1,2],[2,1]])
E = np.array([[1,0,0,0], [0,2,0,0], [0,0,3,0], [0,0,0,4]])

b = np.array([3, 4])
c = np.array([1,1,1,1])
e1 = np.array([1,0,0,0])

test = lambda t: np.array([[0, 2*t], [2*t, 0]])
test1 = np.array([[lambda t: 1, lambda t: np.sin(t)], [lambda t: 2*t, lambda t: 0]])

# integrating lambda nparray element-wise
for i in range(0,4):
    #print(vec(test1)[i](2))
    print(integral(vec(test1)[i],0,np.pi))

# create method to use magnus expansion for time dependent A in Y'=AY