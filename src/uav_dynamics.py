#include the dynamics for the uav

from scipy.linalg import expm
import numpy as np
import math

from scipy.integrate import solve_ivp

MAX_VELOCITY = 20
MIN_VELOCITY = 10
MAX_TURN_RATE = math.pi / 6
MIN_TURN_RATE = -math.pi / 6

TRUTH_MODEL_PROCESS_NOISE = np.array([[0.01, 0.001, 0.0005],
                                      [0.001, 0.01, 0.0005],
                                      [0.0005, 0.0005, 0.001]])

class Dynamical_UAV():

    def __init__(self, initial_state):
        
        #set the initial state
        self.current_state = initial_state

    def update_nominal_state(self, t, x_0, control_nom):
        # currently static method to get nominal state
        # params:
        #   t = current time
        #   x_0 = [xi_0, eta_0, theta_0]
        #   control_nom = v_a and omega, constants for nominal trajectory
        v_a = control_nom[0]
        omega = control_nom[1]

        theta = x_0[2] + omega * t
        xi = x_0[0] + v_a / omega * (np.sin(theta) - np.sin(x_0[2]))
        eta = x_0[1] - v_a / omega * (np.cos(theta) - np.cos(x_0[2]))

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

        if(control[1] > MAX_TURN_RATE):
            print(f"Control Rate Exceeds Bounds: {control[1]}" )
            control[1] = MAX_TURN_RATE

        elif(control[1] < MIN_TURN_RATE):
            print(f"Control Rate Exceeds Bounds: {control[1]}" )
            control[1] = MIN_TURN_RATE

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

        A_nom = jac[0:3, 0:3]
        B_nom = jac[0:3, 3:5]
        # euler approx
        F_k = np.eye(3, 3) + dt * A_nom
        G_k = dt * B_nom

        return F_k, G_k

    def get_process_noise_covariance(self, noise_covarience, dt, control, mapping=np.eye(3)):

        A = (self._get_current_jacobian(self.current_state, control))[0:3, 0:3]

        #use van loan's method the compute the dt process noise matrix Q
        Z = np.vstack([np.hstack([-A, mapping@noise_covarience@np.transpose(mapping)]),
                       np.hstack([np.zeros([3,3]), np.transpose(A)])])
        
        Ze = expm(Z * dt)

        return np.transpose(Ze[3:6,3:6]) * Ze[0:3, 3:6]

    #propagate the current timestep by a timestep dt using the control input control
    def step_nl_propagation(self, control, dt, process_noise=False):
        #Params:
        #   controls = [va, phi_a]
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

            self.current_state = self.current_state + np.random.multivariate_normal(np.zeros([3]), Q)


    def _get_current_jacobian(self, x_nom, control):

        #Params:
        #   controls = [va, phi_a]
        #Returns the linearized jacabian for the UAV
        #[dE, dN, dT] by [dE, dN, dT, dVa, dPhi]
        jac = np.zeros([3,5])

        jac[0][2] = -math.sin(x_nom[2]) * control[0]
        jac[1][2] = math.cos(x_nom[2]) * control[0]
        jac[0][3] = math.cos(x_nom[2])
        jac[1][3] = math.sin(x_nom[2])
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
    
    
if __name__ == "__main__":
    uav = Dynamical_UAV([0,0,math.pi/12])

    print(uav.get_current_jacobian([12, math.pi/12]))

