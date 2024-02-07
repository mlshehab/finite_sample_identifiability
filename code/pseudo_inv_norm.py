import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as alg
from numpy.linalg import norm, pinv
from numpy import sqrt

from scipy.optimize import minimize


m = 4
n = 25
horizon = 4
discount = 0.9


def obj(x):
    
#     m = args[0]
#     n = args[1]
#     horizon = args[2]
#     discount = args[3]
    
    I = np.eye(n)
    sto_mat = []
    
    for i in range(m):
        sto_mat.append(x[i*(n**2):(i+1)*(n**2)].reshape(n,n))
 
    P = np.vstack([p for p in sto_mat])
    E = np.vstack([I for _ in range(m)])
    
    bold_I = np.eye(m*n)
    
    Gamma = np.zeros((horizon*m*n, horizon*n))
    
  
    for i in range(horizon-1):
        Gamma[i*n*m:(i+1)*n*m, i*n:(i+1)*n] = -E
        Gamma[i*n*m:(i+1)*n*m, (i+1)*n:(i+2)*n] =  discount*P
    
    Gamma[ (horizon-1)*m*n:(horizon)*m*n , (horizon-1)*n: horizon*n] = -E
    
    left_eye = np.vstack([bold_I for _ in range(horizon)])
    
    Gamma  = np.hstack((left_eye,Gamma))
    
#     return -np.min(np.linalg.svd(Gamma, compute_uv=False)[:-1])
    return norm(pinv(Gamma))


def generate_stochastic_matrix(n):
    # n: size of the square matrix
    matrix = np.random.rand(n, n)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix

def generate_determ_matrix(n):
    P = np.eye(n)  # 3x3 identity
    [np.random.shuffle(p) for p in P]
    return P  # shuffles rows

def generate_unif_matrix(n):
    P = np.ones((n,n))
    P /= P.sum(axis=1, keepdims=True)
    return P

def choose_random_numbers(m, n):
    return [np.random.randint(0, n-1) for _ in range(m)]

def generate_grid_matrix(n):
    P = np.zeros((n,n))
    for p in P:
        idx = choose_random_numbers(3, n)
        for id_i in idx:
            p[id_i] += 1.0
    P /= P.sum(axis=1, keepdims=True)      
    return P


stochastic_matrices_obj = []
determinis_matrices_obj = []
uniformmmm_matrices_obj = []
gridworldd_matrices_obj = []

trials = 1000

for i in range(trials):
    print(f"Trial: {i}")
    
    x0_s = np.zeros(((n**2)*m,1))
    x0_d = np.zeros(((n**2)*m,1))
    x0_u = np.zeros(((n**2)*m,1))
    x0_g = np.zeros(((n**2)*m,1))

    for i in range(m):
        m_s = generate_stochastic_matrix(n)
        m_d = generate_determ_matrix(n)
        m_u = generate_unif_matrix(n)
        m_g = generate_grid_matrix(n)
        
        x0_s[i*n**2:(i+1)*n**2] = m_s.reshape(n**2,1)
        x0_d[i*n**2:(i+1)*n**2] = m_d.reshape(n**2,1)
        x0_u[i*n**2:(i+1)*n**2] = m_u.reshape(n**2,1)
        x0_g[i*n**2:(i+1)*n**2] = m_g.reshape(n**2,1)
        
    stochastic_matrices_obj.append(obj(x0_s))
    determinis_matrices_obj.append(obj(x0_d))
    uniformmmm_matrices_obj.append(obj(x0_u))
    gridworldd_matrices_obj.append(obj(x0_g))


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5),dpi = 100)

plt.hist(stochastic_matrices_obj, bins = 5, label = 'Stochastic - mean = %.2f'%(np.mean(stochastic_matrices_obj)) )
plt.hist(determinis_matrices_obj, bins = 5, label = 'Determinsi - mean = %.2f'%(np.mean(determinis_matrices_obj)))
plt.hist(uniformmmm_matrices_obj, bins = 5, label = 'Uniformmmm - mean = %.2f'%(np.mean(uniformmmm_matrices_obj)))
plt.hist(gridworldd_matrices_obj,  bins = 5,label = 'GridWorldd - mean = %.2f'%(np.mean(gridworldd_matrices_obj)))
plt.legend()
plt.grid()
plt.xlabel("$||\Gamma^\dagger||$")
plt.ylabel("Counts")
plt.show()