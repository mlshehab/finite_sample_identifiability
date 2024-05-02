import cvxpy as cp
import numpy as np
from GridWorlds import BasicGridWorld

def solve_optimization_problem(gw, mu_0):
    m = gw.n_states
    n = gw.n_actions
    T = gw.horizon
    gamma = gw.discount 

    # construct P and E
    I = np.eye(gw.n_states)
    E = I
    P = gw.transition_probability[:,0,:]

    for a in range(1, gw.n_actions):
        E = np.vstack((E,I))
        P = np.vstack((P, gw.transition_probability[:,a,:]))

    # Define decision variables
    q = [cp.Variable((n, m)) for _ in range(T)]

    # Define the objective function
    objective = 0
    for t in range(T):
        for s in range(n):
            for a in range(m):
                q_t_s_a = q[t][s,a]
                q_t_s_sum = cp.sum(q[t][s,:])   
                objective -= gamma**t *  (cp.log(q_t_s_a) - cp.log(q_t_s_sum))

    # Define constraints

    constraints = []
    # for t in range(1, T):
    #     constraints.append(cp.matmul(P.T,  cp.reshape(q[t-1], shape = (m*n,1), order = 'F')) == cp.matmul(E.T, cp.reshape(q[t], shape = (m*n,1), order = 'F')  ))
    # constraints.append(cp.matmul(E.T,  cp.reshape(q[0], shape = (m*n,1), order = 'F')) == mu_0)
    # constraints.append(cp.sum(cp.vstack(q)) == bar_f)
    for t in range(T):  
        constraints.append(q[t] >= 0)

    # Define and solve optimization problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(verbose=True)

    if prob.status == cp.OPTIMAL:
        # return [q_t.value for q_t in q]
        print("e=here")
    else:
        print("Optimization problem failed to converge.")

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
mu_0 = np.zeros((gw.n_states,1))
mu_0[4] = 1.0

solution = solve_optimization_problem(gw,mu_0)
print("Optimal q values:")
# for t, q_t in enumerate(solution):
#     print(f"t={t}:")
#     print(q_t)
