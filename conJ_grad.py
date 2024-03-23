import numpy as np 
import array_to_latex as a2l 
# Conjugate Gradient method 
def conjugate_gradient(A, b, x0, tol=0.01, max_iter=1000): 
    r = b - np.dot(A, x0) 
    v = r 
    print("v=") 
    a2l.to_ltx(v, frmt = '{:6.2f}', arraytype = 'bmatrix') 
    x = x0
    for i in range(max_iter): 
        print("Iteration",i,":") 
        print("r=") 
        a2l.to_ltx(r, frmt = '{:6.2f}', arraytype = 'bmatrix')
        Av = np.dot(A, v) 
        print("Av=") 
        a2l.to_ltx(Av, frmt = '{:6.2f}', arraytype = 'bmatrix') 
        t = np.dot(r,v) / np.dot(v, Av)
        print("t=",t) 
        x = x + t * v 
        print("x=") 
        a2l.to_ltx(x, frmt = '{:6.2f}', arraytype = 'bmatrix')
        v = r
        r = b - np.dot(A,x)
        print("r=") 
        a2l.to_ltx(r, frmt = '{:6.2f}', arraytype = 'bmatrix')
        rsnew = np.dot(r, r) 
        print("|r|^2=",rsnew) 
        if np.sqrt(rsnew) < tol: 
            break 
        """beta = (rsnew / rsold) 
        print("beta",beta) 
        p = r + beta * p 
        print("new p=") 
        a2l.to_ltx(p, frmt = '{:6.7f}', arraytype = 'bmatrix') 
        rsold = rsnew """
    return x, i+1

A = np.array([[1.00,0.50],[0.50,0.33]])
b = np.array([0.24,0.13])
x0 = np.array([0,0])
solution, iterations = conjugate_gradient(A, b, x0)
print("Solution:", solution)
print("Number of iterations:", iterations)
# Solving by Conjugate Gradient method A = np.array([[0.1,0.2],[0.2,113]]) b = np.array([0.3,113.2]) x0 = np.array([0,0]) solution, iterations = conjugate_gradient(A, b, x0) print("Solution:", solution) print("Number of iterations:", iterations)