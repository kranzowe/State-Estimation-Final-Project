#include the dynamics for the uav

import numpy as np
import math
from scipy.linalg import expm


from scipy.integrate import solve_ivp

MAX_VELOCITY = 3
MIN_VELOCITY = 0
MAX_STEER_ANGLE = 5*math.pi / 12
MIN_STEER_ANGLE = -5*math.pi / 12

class Dynamical_UGV():

    def __init__(self, initial_state):
        
        #set the initial state
        self.current_state = initial_state
        self.L = 0.5

    def get_current_jacobian(self, control):

        #wrap the get jacobian function to enforce limits

        if(control[0] > MAX_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MAX_VELOCITY

        elif(control[0] < MIN_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MIN_VELOCITY

        if(control[1] > MAX_STEER_ANGLE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MAX_STEER_ANGLE

        elif(control[1] < MIN_STEER_ANGLE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MIN_STEER_ANGLE

        return self._get_current_jacobian(control)

    def step_dt_system(self, F, G, control):

        self.current_state = F @ self.current_state + G @ control

        if  self.current_state[2]  > math.pi:
             self.current_state[2] -= 2*math.pi
        elif  self.current_state[2] < -math.pi:
             self.current_state[2] += 2*math.pi

    def state_dt_transition_matrix(self, dt, control, state=None):

        if not (state == None):
            self.current_state = state

        A_hat = np.zeros([5,5])
        A_hat[0:3, :] = self.get_current_jacobian(control)

        F_hat = expm(dt * A_hat)

        return F_hat[0:3, 0:3], F_hat[0:3, 3:5]

    #propagate the current timestep by a timestep dt using the control input control
    def step_jacobian_propagation(self, control, dt):
        #Params:
        #   controls = [vg, phi_g]
        #   dt = scalar intended timestep

        solve_ivp() 

    #propagate the current timestep by a timestep dt using the control input control
    def step_nl_propagation(self, control, dt):
        #Params:
        #   controls = [vg, phi_g]
        #   dt = scalar propagation time

        #solve the ivp 
        result = solve_ivp(self.get_nl_d_state, [0, dt], self.current_state, args=(control,))    

        theta = result.y[2][-1]
        if theta > math.pi:
            theta -= 2*math.pi
        elif theta < -math.pi:
            theta += 2*math.pi
        #update the current system state
        self.current_state = [result.y[0][-1], result.y[1][-1], theta]


    def _get_current_jacobian(self, control):

        #Params:
        #   controls = [vg, phi_g]
        #Returns the linearized jacabian for the UGV
        #[dE, dN, dT] by [dE, dN, dT, dVg, dPhi]
        jac = np.zeros([3,5])

        jac[0][2] = -math.sin(self.current_state[2]) * control[0]
        jac[1][2] = math.cos(self.current_state[2]) * control[0]
        jac[0][3] = math.cos(self.current_state[2])
        jac[1][3] = math.sin(self.current_state[2])
        jac[2][3] = math.tan(control[1]) / self.L
        jac[2][4] = control[0] * (math.tan(control[1])**2 + 1) / self.L

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

        if(control[1] > MAX_STEER_ANGLE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MAX_STEER_ANGLE

        elif(control[1] < MIN_STEER_ANGLE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MIN_STEER_ANGLE

        return self._get_nl_d_state(control)


    def _get_nl_d_state(self, control):
        
        #Params:
        #   controls = [vg, phi_g]
        #Returns the d-state vector for the "current state" and given control vector
        # [dE, dN, dT]

        d_state = np.zeros([3])

        d_state[0] = control[0] * math.cos(self.current_state[2])
        d_state[1] = control[0] * math.sin(self.current_state[2])
        d_state[2] = (control[0]/self.L) * math.tan(control[1])

        return d_state
    
    


