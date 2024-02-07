import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as alg
from numpy.linalg import norm, pinv
from numpy import sqrt

from scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')


def reduce_rank(matrix, m):
    # Perform Singular Value Decomposition (SVD)
    U, Sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Set the last m singular values to zero
    Sigma[-m:] = 0
    
    # Reconstruct the matrix with reduced rank
    reduced_matrix = U @ np.diag(Sigma) @ Vt
    
    return reduced_matrix


if __name__ == '__main__':

    m = 100
    n = 20 

    A = np.random.randn(m,n)
    b = np.random.randn(m)
    A = reduce_rank(A,10)

    sol = alg.lstsq(A,b)
    x0 = sol[0]
    print("r = %.2f"% norm(A@x0-b))
    eps = 1
    res_ = []

    for i in range(1000):
        d_b = np.random.random(m)
        d_b = (1/np.linalg.norm(d_b))*d_b
        delta_b = eps*d_b
        
        sol_ = alg.lstsq(A, b + delta_b)
        # print(norm(sol_[0]))
        res_.append(norm(x0 - sol_[0]))

    ub = np.linalg.norm(np.linalg.pinv(A))*np.linalg.norm(delta_b)
    y = pinv(A@A.T)
    # plt.figure(figsize=(10,5),dpi = 100)
    plt.hist(res_)
    plt.axvline(x = ub, color = 'r', linestyle = '-',label = "$||A^+||* ||\delta b||$ = %.3f"%(ub) ) 
    plt.axvline(x = min(res_), color = 'k', linestyle = '--', label = "min res = %.3f"%(min(res_))) 
    plt.axvline(x = max(res_), color = 'g', linestyle = '--', label = "max res = %.3f"%(max(res_)))
    plt.xlabel('Error Norm $||\delta x||$')
    plt.title("$||\delta b||$= %.1f , A is %d by %d and rank(A) = %d"%(eps,m,n, np.linalg.matrix_rank(A)))

    plt.show()




    