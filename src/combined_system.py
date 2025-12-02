
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

    def get_dt_state_transition_matrices(self, dt, x_nom, control_nom):

        F = np.zeros([6,6])
        G = np.zeros([6,4])

        f_ugv, g_ugv = self.ugv.state_dt_transition_matrix(dt, x_nom[0:3], control_nom[0:2])
        f_uav, g_uav = self.uav.state_dt_transition_matrix(dt, x_nom[3:6], control_nom[2:4])

        F[0:3, 0:3] = f_ugv
        F[3:6, 3:6] = f_uav
        G[0:3, 0:2] = g_ugv
        G[3:6, 2:4] = g_uav

        return F, G
    
    def get_dt_H_and_Omega(self, dt, x_nom, control_nom):

        # simplifying relations
        dx = x_nom[3] - x_nom[0]
        dy = x_nom[4] - x_nom[1]
        rho = np.sqrt(dx**2 + dy**2)
        rho2 = dx**2 + dy**2

        # cet CT=DT jacobians for the measurement matrix
        H = np.zeros((5, 6))
        H[0, 0] = dy / rho2
        H[0, 1] = -dx / rho2
        H[0, 2] = -1.0
        H[0, 3] = -dy / rho2
        H[0, 4] = dx / rho2

        H[1, 0] = -dx / rho
        H[1, 1] = -dy / rho
        H[1, 3] = dx / rho
        H[1, 4] = dy / rho

        H[2, 0] = dy / rho2
        H[2, 1] = -dx / rho2
        H[2, 3] = -dy / rho2
        H[2, 4] = dx / rho2
        H[2, 5] = -1.0

        H[3, 3] = 1.0
        H[4, 4] = 1.0

        Omega = np.eye(6)

        return H, Omega

    def step_dt_states(self, F, G, control_perturb):

        self.ugv.step_dt_system(F[0:3, 0:3], G[0:3, 0:2], control_perturb[0:2])
        self.uav.step_dt_system(F[3:6, 3:6], G[3:6, 2:4], control_perturb[2:4])
        
        #update the current system state
        self.current_state = np.hstack([self.ugv.current_state, self.uav.current_state])



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

        theta_g = result.y[2][-1]
        if theta_g > math.pi:
            theta_g -= 2*math.pi
        elif theta_g < -math.pi:
            theta_g += 2*math.pi
        theta_a = result.y[5][-1]
        if theta_a > math.pi:
            theta_a -= 2*math.pi
        elif theta_a < -math.pi:
            theta_a += 2*math.pi

        #update the current system state
        self.current_state = [result.y[0][-1], result.y[1][-1], theta_g, result.y[3][-1], result.y[4][-1], theta_a]

    def finite_difference_F(self):
        

    def create_measurements_from_states(self):

        measurement_array = np.zeros([5])

        #bearing to the auv to the ugv
        measurement_array[0] = math.atan2(self.current_state[4] - self.current_state[1], self.current_state[3] - self.current_state[0]) - self.current_state[2]

        #positional distance
        measurement_array[1] = math.sqrt((self.current_state[0] - self.current_state[3]) ** 2 + (self.current_state[1] - self.current_state[4]) ** 2)

        #bearing from the auv to the ugv
        measurement_array[2] = math.atan2(self.current_state[1] - self.current_state[4], self.current_state[0] - self.current_state[3]) - self.current_state[5]

        #uav position
        measurement_array[4] = self.current_state[4]
        measurement_array[3] = self.current_state[3]

        #normalize angles
        if measurement_array[0] > math.pi:
            measurement_array[0] -= 2*math.pi
        elif measurement_array[0] < -math.pi:
            measurement_array[0] += 2*math.pi
        if measurement_array[2] > math.pi:
            measurement_array[2] -= 2*math.pi
        elif measurement_array[2] < -math.pi:
            measurement_array[2] += 2*math.pi

        return measurement_array

    def create_dt_measurements(self, x_nom):
        full_state = x_nom
        measurement_nom_array = np.zeros([5])

        # bearing to the auv to the ugv
        measurement_nom_array[0] = math.atan2(full_state[4] - full_state[1], full_state[3] - full_state[0]) - full_state[2]

        # positional distance
        measurement_nom_array[1] = math.sqrt((full_state[0] - full_state[3]) ** 2 + (full_state[1] - full_state[4]) ** 2)

        # bearing from the auv to the ugv
        measurement_nom_array[2] = math.atan2(full_state[1] - full_state[4], full_state[0] - full_state[3]) - full_state[5]

        # uav position
        measurement_nom_array[4] = full_state[4]
        measurement_nom_array[3] = full_state[3]

        # normalize angles
        if measurement_nom_array[0] > math.pi:
            measurement_nom_array[0] -= 2 * math.pi
        elif measurement_nom_array[0] < -math.pi:
            measurement_nom_array[0] += 2 * math.pi
        if measurement_nom_array[2] > math.pi:
            measurement_nom_array[2] -= 2 * math.pi
        elif measurement_nom_array[2] < -math.pi:
            measurement_nom_array[2] += 2 * math.pi

        # simplifying relations
        dx = x_nom[3] - x_nom[0]
        dy = x_nom[4] - x_nom[1]
        rho = np.sqrt(dx**2 + dy**2)
        rho2 = dx**2 + dy**2

        # cet CT=DT jacobians for the measurement matrix
        H = np.zeros((5, 6))
        H[0, 0] = dy / rho2
        H[0, 1] = -dx / rho2
        H[0, 2] = -1.0
        H[0, 3] = -dy / rho2
        H[0, 4] = dx / rho2

        H[1, 0] = -dx / rho
        H[1, 1] = -dy / rho
        H[1, 3] = dx / rho
        H[1, 4] = dy / rho

        H[2, 0] = dy / rho2
        H[2, 1] = -dx / rho2
        H[2, 3] = -dy / rho2
        H[2, 4] = dx / rho2
        H[2, 5] = -1.0

        H[3, 3] = 1.0
        H[4, 4] = 1.0

        measurement_perturbation_array = H @ self.current_state

        return measurement_nom_array + measurement_perturbation_array


    def _get_current_jacobian(self, control):

        return


    def get_nl_d_state(self, t = None, state = None, control = None):

        if not (state is None):
            #update the system to a given state if one is provided
            self.current_state = state

        if(control is None):
            control = np.zeros(4,1)

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

        control_ugv = control[0:2]  # [vg, phi_g]
        control_uav = control[2:4]  # [va, omega_a]
        
        # Update individual vehicle states from combined state
        self.ugv.current_state = self.current_state[0:3]
        self.uav.current_state = self.current_state[3:6]

        #step the vehicles with the corrected controls
        d_state_uav = self.uav._get_nl_d_state(control_uav)
        d_state_ugv = self.ugv._get_nl_d_state(control_ugv)

        return np.hstack((d_state_ugv, d_state_uav))
    
