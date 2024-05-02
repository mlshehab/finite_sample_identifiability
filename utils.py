import math
import numpy as np
import scipy.special
import cvxpy as cp

inf  = float("inf")

def soft_bellman_operation(env, reward):
    
    # Input:    env    :  environment object
    #           reward :  SxA dimensional vector 
    #           horizon:  finite horizon
        
    discount = env.discount
    horizon = env.horizon
    
    if horizon is None or math.isinf(horizon):
        raise ValueError("Only finite horizon environments are supported")
    
    n_states  = env.n_states
    n_actions = env.n_actions
    
#     T = env.transition_matrix
    
    V = np.zeros(shape = (horizon, n_states))
    Q = np.zeros(shape = (horizon, n_states,n_actions))
        
    broad_R = reward

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = broad_R
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
#         next_values_s_a = T @ V[t + 1, :]
#         next_values_s_a = next_values_s_a.reshape(n_states,n_actions)
        for a in range(n_actions):
            Ta = env.transition_probability[:,a,:]
            next_values_s_a = Ta@V[t + 1, :]
            Q[t, :, a] = broad_R[:,a] + discount * next_values_s_a
            
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi


def create_Pi(pi):
    horizon, n_states, n_actions = pi.shape
    
    Pi = np.zeros(((horizon - 1)*n_states*n_actions,1))
    
    for t in range(horizon-1):
        curr_pi = pi[t].flatten('F')[:,None]
#         print(curr_pi.shape)
        next_pi = pi[t+1].flatten('F')[:,None]
        Pi[t*n_states*n_actions:(t+1)*n_states*n_actions] = np.log(curr_pi) - np.log(next_pi)
    return Pi


