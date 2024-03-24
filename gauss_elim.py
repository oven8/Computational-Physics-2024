import numpy as np 
import array_to_latex as a2l 
# Conjugate Gradient method 
def gauss_elim(A, b):
    n = len(A)
    
    # Augmenting the matrix
    AMN = np.hstack((A, b.reshape(-1, 1)))
    
    print("A=")
    a2l.to_ltx(AMN[0:,0:n], frmt = '{:6.7f}', arraytype = 'bmatrix')
    print("B=")
    a2l.to_ltx(AMN[0:,n], frmt = '{:6.7f}', arraytype = 'bmatrix')
    
    
    
    for i in range(n):
        # Partial pivoting
        max_index = np.argmax(np.abs(AMN[i:, i])) + i
        if max_index != i:
            AMN[[i, max_index]] = AMN[[max_index, i]]
            print("After partial pivoting:")
            print("R",i+1,"swap R",max_index+1)
            print("A=")
            a2l.to_ltx(AMN[0:,0:n], frmt = '{:6.7f}', arraytype = 'bmatrix')
            print("B=")
            a2l.to_ltx(AMN[0:,n], frmt = '{:6.7f}', arraytype = 'bmatrix')
            
        #Normalization
        print("R",i+1,"= R",i+1,"/",AMN[i,i])
        AMN[i] = AMN[i]/AMN[i,i]
        print("A=")
        a2l.to_ltx(AMN[0:,0:n], frmt = '{:6.7f}', arraytype = 'bmatrix')
        print("B=")
        a2l.to_ltx(AMN[0:,n], frmt = '{:6.7f}', arraytype = 'bmatrix')
            
        # Elimination
        for j in range(i+1, n):
            print("R",j+1,"= R",j+1,"-",AMN[j,i],"* R",i+1)
            AMN[j, i:] -= AMN[j, i] * AMN[i, i:]
            print("A=")
            a2l.to_ltx(AMN[0:,0:n], frmt = '{:6.7f}', arraytype = 'bmatrix')
            print("B=")
            a2l.to_ltx(AMN[0:,n], frmt = '{:6.7f}', arraytype = 'bmatrix')
                
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = (AMN[i, -1] - np.dot(AMN[i, i+1:n], x[i+1:]))
            
    return x

# Example usage:
A = np.array([[0.1,0.2],[0.2,113]]).astype(float)
b = np.array([0.3,113.2]).astype(float)

solution = gauss_elim(A, b)
print("Solution:", solution)
"""A = np.array([[0.1,0.2],[0.2,113]])
b = np.array([0.3,113.2])
x0 = np.array([0,0])
solution, iterations = conjugate_gradient(A, b, x0)
print("Solution:", solution)
print("Number of iterations:", iterations)"""