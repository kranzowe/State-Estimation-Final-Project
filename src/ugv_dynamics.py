#include the dynamics for the uav

import numpy as np
import math
from scipy.linalg import expm


from scipy.integrate import solve_ivp

MAX_VELOCITY = 3
MIN_VELOCITY = 0
MAX_STEER_ANGLE = 5*math.pi / 12
MIN_STEER_ANGLE = -5*math.pi / 12

TRUTH_MODEL_PROCESS_NOISE = np.array([[0.001, 0.0, 0.0],
                                      [0, 0.001, 0.0],
                                      [0, 0.0, 0.01]])

class Dynamical_UGV():

    def __init__(self, initial_state):
        
        #set the initial state
        self.current_state = initial_state
        self.L = 0.5

    def update_nominal_state(self, t, x_0, control_nom):
        # get nominal state assuming constant turn
        # params:
        #   t = current time
        #   x_0 = [xi_0, eta_0, theta_0]
        #   control_nom = v_g and phi, constants for nominal trajectory
        v_g = control_nom[0]
        phi = control_nom[1]
        omega = v_g / self.L * np.tan(phi)
        theta = x_0[2] + omega * t
        xi = x_0[0] + v_g / omega * (np.sin(theta) - np.sin(x_0[2]))
        eta = x_0[1] - v_g / omega * (np.cos(theta) - np.cos(x_0[2]))

        if theta > math.pi:
            while theta > math.pi:
                theta -= 2*math.pi
        elif theta < -math.pi:
            while theta < -math.pi:
                theta += 2*math.pi

        nom_state = np.array([xi, eta, theta])
        return nom_state

    def get_current_jacobian(self, x_nom, control):

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

        return self._get_current_jacobian(x_nom, control)

    def step_dt_system(self, F, G, control_perturb):

        self.current_state = F @ self.current_state + G @ control_perturb

        if  self.current_state[2]  > math.pi:
             self.current_state[2] -= 2*math.pi
        elif  self.current_state[2] < -math.pi:
             self.current_state[2] += 2*math.pi

    def state_dt_transition_matrix(self, dt, x_nom, control_nom, state=None):

        if not (state == None):
            self.current_state = state

        jac = self.get_current_jacobian(x_nom, control_nom)

        A_nom = jac[0:3,0:3]
        B_nom = jac[0:3,3:5]
        # euler approx
        F_k = np.eye(3,3) + dt*A_nom
        G_k = dt*B_nom

        return F_k, G_k

    #propagate the current timestep by a timestep dt using the control input control
    def step_nl_propagation(self, control, dt, process_noise=False):
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

        if(process_noise):
            Q = self.get_process_noise_covariance(TRUTH_MODEL_PROCESS_NOISE, dt, control)

            self.current_state = self.current_state + np.linalg.cholesky(Q) @ np.random.multivariate_normal(np.zeros([3]), np.eye(3))




    def _get_current_jacobian(self, x_nom, control):

        #Params:
        #   controls = [vg, phi_g]
        #Returns the linearized jacabian for the UGV
        #[dE, dN, dT] by [dE, dN, dT, dVg, dPhi]
        jac = np.zeros([3,5])

        jac[0][2] = -math.sin(x_nom[2]) * control[0]
        jac[1][2] = math.cos(x_nom[2]) * control[0]
        jac[0][3] = math.cos(x_nom[2])
        jac[1][3] = math.sin(x_nom[2])
        jac[2][3] = math.tan(control[1]) / self.L
        jac[2][4] = control[0] * (math.tan(control[1])**2 + 1) / self.L

        return jac


    def get_process_noise_covariance(self, noise_covarience, dt, control, mapping=np.eye(3)):

        # A = (self._get_current_jacobian(self.current_state, control))[0:3, 0:3]

        # #use van loan's method the compute the dt process noise matrix Q
        # Z = np.vstack([np.hstack([-A, mapping@noise_covarience@np.transpose(mapping)]),
        #                np.hstack([np.zeros([3,3]), np.transpose(A)])])
        
        # Ze = expm(Z * dt)

        # return np.transpose(Ze[3:6,3:6]) * Ze[0:3, 3:6]

        return noise_covarience


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
    
    


