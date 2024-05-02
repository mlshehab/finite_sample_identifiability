from scipy.optimize import minimize
from GridWorlds import BasicGridWorld
import numpy as np
from utils import soft_bellman_operation
from scipy.optimize import LinearConstraint, Bounds

def q2x(q):
    return np.ravel(q)

def x2q(x,T,m,n):
    # q is of shape T by m by n, meaning that it has m rows and n columns
    return x.reshape((T,m,n))

def policy2q(pi):
    T, n,m = pi.shape
    q = np.zeros((T,m,n))

    for t in range(T):
        for s in range(n):
            for a in range(m):
                q[t,a,s] = pi[t,s,a]/np.sum(pi[t,s,:])
    return q


def objective_function(x, *args):
    # Unpack additional arguments
    gw = args[0]

    n = gw.n_states
    m = gw.n_actions
    T = gw.horizon
    gamma = gw.discount 

    q = x2q(x,T,m,n)    
    # Define the objective function
    objective = 0
    for t in range(T):
        for s in range(n):
            for a in range(m): 
                objective -= gamma**t * q[t][a,s]* (np.log(q[t][a,s]/ np.sum(q[t][:,s]))) 

    return objective

# construct A@x == b equality constraint
def construct_A_and_b(gw,bar_f):
    horizon = gw.horizon
    m = gw.n_actions
    n = gw.n_states
    mu_0 = gw.mu_0

    # construct P and E
    I = np.eye(gw.n_states)
    E = I
    P = gw.transition_probability[:,0,:]

    for a in range(1, gw.n_actions):
        E = np.vstack((E,I))
        P = np.vstack((P, gw.transition_probability[:,a,:]))
    ########################################################

    cols_ = horizon*m*n
    rows_ = (horizon-1)*n 
    A_I =  np.zeros( (rows_, cols_))
    b_I = np.zeros((rows_,1))

    for i in range(horizon-1):   
        A_I[n*i:n*(i+1),m*n*i:m*n*(i+1)]= P.T 
        A_I[n*i:n*(i+1),m*n*(i+1):m*n*(i+2)]= -E.T

    F = gw.F_matrix().T
    A_II = np.tile(F, (1,horizon))
    b_II = bar_f

    A_III = np.hstack((-E.T, np.zeros((n,(horizon-1)*m*n))))
    b_III = mu_0

    A = np.vstack((A_I, A_II, A_III))
    b = np.vstack((b_I, b_II, b_III))

    return A,b


# find a policy and compute its bar_f
# Example usage:
grid_size = 5
wind = 0.1
discount = 0.9   
start_state = [i for i in range(25)]
feature_dim = 2

landmark_locations = [(0,24), (3,1), (3,19), (14,14)]
p1,p2 = landmark_locations[-1]
horizon = 3
theta = 5*np.ones((2,1))
theta[1] -= 5

gw = BasicGridWorld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta,"dense")
m = gw.n_actions
n= gw.n_states
F = gw.F_matrix().T

mu_0 = np.zeros((gw.n_states,1))
mu_0[4] = 1.0

gw.mu_0 = mu_0 
x0 = (1/(m*n))*np.ones((horizon*gw.n_states*gw.n_actions,1))

reward = gw.reward_v
V,Q,pi = soft_bellman_operation(gw,reward)


q = policy2q(pi)
x = q2x(q)

bar_f =  np.tile(F, (1,horizon))@x[:,np.newaxis]


A,b = construct_A_and_b(gw,bar_f)


def con(x, *args):
    A,b = args
    return (A@x - b)[0]

# cons = [ {'type': 'ineq', 'fun': lambda x: x[i]} for i in range(horizon*m*n)]
# cons += [{'type':'eq', 'fun':con, 'args':(A,b)}]
# bnd = Bounds(lb = 0, ub = np.inf)
# cons = (LinearConstraint(A,lb = b.ravel(), ub = b.ravel()) , {'type': 'ineq', 'fun': lambda x: x})

eps = 1e-4
# print(eps*np.ones(x0.shape[0]).shape)
cons = [LinearConstraint(A,lb = b.ravel(), ub = b.ravel()),\
      LinearConstraint(A = np.eye(x0.shape[0]), lb = 0*np.ones(x0.shape[0]), ub = +np.inf  )]

res = minimize(objective_function, x0 = x0.ravel(), args = gw, method ='trust-constr', constraints= cons, options={'gtol': 1e-6, 'disp': True})

