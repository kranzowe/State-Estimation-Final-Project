
import numpy as np
import math

from scipy.integrate import solve_ivp


MAX_UGV_VELOCITY = 3
MIN_UGV_VELOCITY = 0
MAX_STEER_ANGLE = 5*math.pi / 12
MIN_STEER_ANGLE = -5*math.pi / 12


MAX_UAV_VELOCITY = 20
MIN_UAV_VELOCITY = 10
MAX_TURN_RATE = math.pi / 6
MIN_TURN_RATE = -math.pi / 6

class CombinedSystem():

    def __init__(self, ugv, uav):
        self.uav = uav
        self.ugv = ugv
        self.current_state = np.hstack((ugv.current_state, uav.current_state))

    def get_current_jacobian(self, control):

        return

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
        result = solve_ivp(self.get_nl_d_state, [0, dt], self.current_state, args=(control,))    


        theta_g= result.y[2][-1]
        if theta_g > math.pi:
            theta_g -= 2*math.pi
        elif theta_g < -math.pi:
            theta_g += 2*math.pi
        theta_a= result.y[5][-1]
        if theta_a > math.pi:
            theta_a -= 2*math.pi
        elif theta_a < -math.pi:
            theta_a += 2*math.pi
        #update the current system state
        self.current_state = [result.y[0][-1], result.y[1][-1], theta_g, result.y[3][-1], result.y[4][-1], theta_a]


    def _get_current_jacobian(self, control):

        return


    def get_nl_d_state(self, t = None, state = None, control = None):

        if not (state is None):
            #update the system to a given state if one is provided
            self.current_state = state

        if(control is None):
            control = np.zeros(2,1)

        #wrap the nl d_state function to enforce limits
        if(control[0] > MAX_UGV_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MAX_UGV_VELOCITY

        elif(control[0] < MIN_UGV_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[0]}" )
            control[0] = MIN_UGV_VELOCITY

        if(control[1] > MAX_STEER_ANGLE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MAX_STEER_ANGLE

        elif(control[1] < MIN_STEER_ANGLE):
            print(f"Control Velocity Exceeds Bounds: {control[1]}" )
            control[1] = MIN_STEER_ANGLE

        if(control[2] > MAX_UAV_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[2]}" )
            control[2] = MAX_UAV_VELOCITY

        elif(control[2] < MIN_UAV_VELOCITY):
            print(f"Control Velocity Exceeds Bounds: {control[2]}" )
            control[2] = MIN_UAV_VELOCITY

        if(control[3] > MAX_TURN_RATE):
            print(f"Control Velocity Exceeds Bounds: {control[3]}" )
            control[3] = MAX_TURN_RATE

        elif(control[3] < MIN_TURN_RATE):
            print(f"Control Velocity Exceeds Bounds: {control[3]}" )
            control[3] = MIN_TURN_RATE

        return self._get_nl_d_state(control)


    def _get_nl_d_state(self, control):
        
        #Params:
        #   controls = [va, phi_a]
        #Returns the d-state vector for the "current state" and given control vector
        # [dE, dN, dT]

        d_state_uav = self.uav._get_nl_d_state(control)
        d_state_ugv = self.ugv._get_nl_d_state(control)

        return np.hstack((d_state_ugv, d_state_uav))