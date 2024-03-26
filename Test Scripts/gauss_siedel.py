import numpy as np 
import array_to_latex as a2l 

def gauss_siedel(A, b, x0, tol=0.01, max_iter=100):
    n = len(A)
    print("A=")
    a2l.to_ltx(A, frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("B=")
    a2l.to_ltx(b, frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("x0=")
    a2l.to_ltx(x0, frmt = '{:6.7f}', arraytype = 'bmatrix')
    U = -np.triu(A,k=1)
    L = -np.tril(A,k=-1)
    D = A + U + L
    T = np.matmul(np.linalg.inv(np.add(D,-L)),U)
    C = np.matmul(np.linalg.inv(np.add(D,-L)),b)
    print("D=")
    a2l.to_ltx(D, frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("L=")
    a2l.to_ltx(L, frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("U=")
    a2l.to_ltx(U, frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("T=")
    a2l.to_ltx(T, frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("C=")
    a2l.to_ltx(C, frmt = '{:6.7f}', arraytype = 'bmatrix')
    x = x0[:]
    for k in range(max_iter):
        print("iteration=",k)
        x_old = x
        x = np.add(np.matmul(T,x),C)
        print("x=")
        a2l.to_ltx(x, frmt = '{:6.7f}', arraytype = 'bmatrix')
        if abs(x - x_old).all() < tol:
            return x
    raise ValueError("Gauss-Seidel method did not converge")
    
A = np.array([[4, -1, 0],
     [-1, 4, -1],
     [0, -1, 3]]).astype(float)
b = np.array([5, -10, 15]).astype(float)
x0 = np.array([0, 0, 0]).astype(float)
solution = gauss_siedel(A, b, x0)
print("Solution:", solution)