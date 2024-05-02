import cvxpy as cp

# Define variables
T = 10  # Example value of T
gamma = 0.9  # Example value of gamma
S = 5  # Example number of states
A = 3  # Example number of actions

q = [cp.Variable((S, A)) for _ in range(T)]
# Objective function
objective = 0
for t in range(T):
    for s in range(S):
        q_sum = cp.sum(q[t][s, :])
        for a in range(A):
            q_sa = q[t, s, a]
            q_sap = q[t][ s, :]
            kl_div = cp.kl_div(q_sap, q_sa * cp.inv_pos(q_sum))
            objective += gamma ** t * q_sa * kl_div

# Minimize objective
problem = cp.Problem(cp.Minimize(-objective))

# Check if problem is DCP
print("Is problem DCP:", problem.is_dcp())
