from turtle import forward
from badMethods import *
from methods import *

A = np.array([[1,2],[3,4]])
h = 0.0001

print(backward_euler_exp(A, h))
print(sp.linalg.expm(A))