from scipy.optimize import minimize
from GridWorlds import BasicGridWorld
import numpy as np
from main_scipy import objective_function
from numpy import abs

# Example usage:
grid_size = 5
wind = 0.1
discount = 1   
start_state = [i for i in range(25)]
feature_dim = 2

landmark_locations = [(0,24), (3,1), (3,19), (14,14)]
p1,p2 = landmark_locations[-1]
horizon = 3
theta = 5*np.ones((2,1))
theta[1] -= 5

gw = BasicGridWorld(grid_size,wind,discount,horizon,start_state, feature_dim, p1,p2,theta,"dense")
m = gw.n_actions
n = gw.n_states
mu_0 = np.zeros((gw.n_states,1))
mu_0[4] = 1.0

gw.mu_0 = mu_0 


i = 1
print(f"Running Assertion #{i} ...")
assert abs(objective_function(np.ones((horizon*gw.n_states*gw.n_actions,1)),gw) - horizon*m*n*np.log(m) ) <= 1e-5
print(f"Passed Assertion #{i}!!!")
i+=1

print(f"Running Assertion #{i} ...")
assert abs(objective_function((1/(m*n))*np.ones((horizon*gw.n_states*gw.n_actions,1)),gw) - horizon*np.log(m) ) <= 1e-5
print(f"Passed Assertion #{i}!!!")
i+=1