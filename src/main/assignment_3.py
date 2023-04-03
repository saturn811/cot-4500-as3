#Kendrick Dawkins
#Assignment 3

import numpy
import numpy as np
from numpy import array, zeros
import scipy
import scipy.linalg
import scipy.linalg as lg
np.set_printoptions(precision=7, suppress=True, linewidth=100)

#QUESTION 1)

def f(x,y):
    return x - (y**2)

def euler(x0, y0, xn, n):

    h = (xn-x0) / n

    
    for i in range(n):
        slope = f(x0, y0)
        yn = y0 + h * slope
        y0 = yn
        x0 = x0+h

    print('%.5f\n' %(yn))


x0 = 0
y0 = 1
xn = 2
step = 10


euler(x0,y0,xn,step)
    

#QUESTION 2)

def f(x,y):
    return x - (y**2)

def rk4(x0, y0, xn, n):
    h = (xn-x0)/n
    n = 10
    
    for i in range(n):
        k1 = h * (f(x0, y0))
        k2 = h * (f((x0+h/2), (y0+k1/2)))
        k3 = h * (f((x0+h/2), (y0+k2/2)))
        k4 = h * (f((x0+h), (y0+k3)))
        k = (k1+2*k2+2*k3+k4)/6
        yn = y0 + k
        y0 = yn
        x0 = x0+h

   
    print('%.5f\n' %(yn))

x0 = 0
y0 = 1
xn = 2
step = 10

rk4(x0,y0,xn,step)


#QUESTION 3)

a = array([[2, -1, 1],
           [1, 3, 1],
           [-1, 5, 4],], float)
b = array([6, 0, -3], float)
n = len(b)
x = zeros(n, float)


for k in range(n-1):
    for i in range(k+1, n):
        fctr = a[i, k] / a[k, k]
        for j in range(k, n):
            a[i, j] = a[i, j] - fctr*a[k, j]
        b[i] = b[i] - fctr*b[k]


x[n-1] = b[n-1] / a[n-1, n-1]
for i in range(n-2, -1, -1):
    Sum = b[i]
    for j in range(i+1, n):
        Sum = Sum - a[i, j]*x[j]
    x[i] = Sum / a[i, i]


print(x, '\n')

#QUESTION 4a)

A = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype=np.double)
det = np.linalg.det(A)
print('%.5f\n' %(det))

#QUESTION 4b)


P, L, U = lg.lu(A)

print(L, '\n')

#QUESTION 4c)

print(U, '\n')

#QUESTION 5)

Q = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])


def diagonally_dominant(Q):
    z = np.diag(np.abs(Q)) 
    j = np.sum(np.abs(Q), axis=1) - z 
    if np.all(z > j):
        print ('True\n')
    else:
        print ('False\n')
    return

diagonally_dominant(Q)

#QUESTION 6)

g = np.array([[2,2,1], [2,3,0], [1,0,2]])

def positive_definite(g):
    if np.all(np.linalg.eigvals(g) > 0):
        print('True')
    else:
        print('False')
    return

positive_definite(g)
