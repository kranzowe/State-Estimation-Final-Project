#include the dynamics for the uav

import numpy as np
import math

class Dynamical_UAV():

    #current state of the uav
    current_state = np.zeros([3,1])

    def __init__(self, initial_state):
        
        #set the initial state
        current_state = initial_state

    def get_current_jacobian(self, control):

        #Params:
        #   controls = [va, phi_a]
        #Returns the linearized jacabian for the UAV
        #[dE, dN, dT] by [dE, dN, dT, dVa, dPhi]
        jac = np.zeros([3,5])

        jac[0][2] = -math.sin(self.current_state[2]) * control[0]
        jac[1][2] = math.cos(self.current_state[2]) * control[0]
        jac[2][3] = math.cos(self.current_state[2])
        jac[2][3] = math.sin(self.current_state[2])
        jac[2][4] = 1

        return jac

