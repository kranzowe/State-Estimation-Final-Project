#include the dynamics for the uav

import numpy as np
import math

from scipy.integrate import solve_ivp

MAX_VELOCITY = 20
MIN_VELOCITY = 10
MAX_TURN_RATE = math.pi / 6
MIN_TURN_RATE = -math.pi / 6

class Dynamical_UAV():

    #current state of the uav
    current_state = np.zeros([3])

    def __init__(self, initial_state):
        
        #set the initial state
        current_state = initial_state

    def get_current_jacobian(self, control):

        #wrap the get jacobian function to enforce limits

        if(control[0] > MAX_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MAX_VELOCITY

        elif(control[0] < MIN_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MIN_VELOCITY

        if(control[1] > MAX_TURN_RATE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MAX_TURN_RATE

        elif(control[1] < MIN_TURN_RATE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MIN_TURN_RATE

        return self._get_current_jacobian(control)

    #propagate the current timestep by a timestep dt using the control input control
    def step_jacobian_propagation(self, control, dt):
        #Params:
        #   controls = [va, phi_a]
        #   dt = scalar intended timestep

        solve_ivp() 

    #propagate the current timestep by a timestep dt using the control input control
    def step_nl_propagation(self, control, dt):
        #Params:
        #   controls = [va, phi_a]
        #   dt = scalar propagation time

        #solve the ivp 
        result = solve_ivp(self.get_nl_d_state, [0, dt], self.current_state, args=[control])    

        #update the current system state
        self.current_state = [result.y[0][-1], result.y[1][-1], result.y[2][-1]]


    def _get_current_jacobian(self, control):

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


    def get_nl_d_state(self, t = None, state = None, control = None):

        if not (state is None):
            #update the system to a given state if one is provided
            self.current_state = state

        if(control is None):
            control = np.zeros(2,1)

        #wrap the nl d_state function to enforce limits

        if(control[0] > MAX_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MAX_VELOCITY

        elif(control[0] < MIN_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MIN_VELOCITY

        if(control[1] > MAX_TURN_RATE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MAX_TURN_RATE

        elif(control[1] < MIN_TURN_RATE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MIN_TURN_RATE

        return self._get_nl_d_state(control)


    def _get_nl_d_state(self, control):
        
        #Params:
        #   controls = [va, phi_a]
        #Returns the d-state vector for the "current state" and given control vector
        # [dE, dN, dT]

        d_state = np.zeros([3])

        d_state[0] = control[0] * math.cos(self.current_state[2])
        d_state[1] = control[0] * math.sin(self.current_state[2])
        d_state[2] = control[1]

        return d_state
    
    


