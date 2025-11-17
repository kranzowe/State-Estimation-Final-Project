'''File for dynamics of the system'''

import numpy as np
from scipy.integrate import solve_ivp

class StateVector():
    def __init__(self, zeta, eta, theta, is_ugv):
        self.is_ugv = is_ugv # uav assumed when false
        self.state = np.array([zeta, eta, theta])


class UGV():
    def __init__(self, state):
        self.state = state
        self.l = 0.1
    
    def propagate(self, t_evals, u):

        if self.is_ugv:
            sol = solve_ivp(state_deriv_ugv, (min(t_evals), max(t_evals) + 0.1), self.state.state, t_eval=t_evals, args=(u, self.l))
        else:
            #sol = solve_ivp(state_deriv_uav, (min(t_evals), max(t_evals) + 0.1), t_eval=t_evals, args= u)
            print('hi')

        
def state_deriv_ugv(t, y, u, l):
    dy = np.zeros((3,))
    dy[0] = u[0] * np.cos(y[2]) # TODO add w
    dy[1] = u[0] * np.sin(y[2])
    dy[2] = u[0]/l * np.tan(u[1])
    return dy