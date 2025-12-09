''' Filter classes of project (LKF and EKF)'''

import numpy as np
from copy import deepcopy

# todo: bar matrices need k subscript and need to be evaled at x_star, u_star at each step before update and correct
class LKF():
    def __init__(self, dt, 
                 nominal_ephem, 
                 nominal_controls,
                 nominal_measurements,
                 combined_system,
                 Q, 
                 R,
                 P_0,
                 dx_0):
        # dt linearized system information
        self.dt = dt
        self.nominal_ephem = nominal_ephem
        self.nominal_controls = nominal_controls
        self.nominal_measurements = nominal_measurements
        self.combined_system = combined_system

        # process and measurement noise
        self.Q = Q
        self.R = R

        self.Kk = None
        # current values for estimated covariance and state disturbance
        self.dx_pre = np.zeros((6,))
        self.dx_post = dx_0 # set post initially, since this is what pre will prop
        self.P_pre = np.zeros((6,6))
        self.P_post = P_0

        # adding control, even though control is constant.
        # these should stay at 0, and not update
        self.du_prev = np.zeros((4,))
        self.du = np.zeros((4,))

        # ephemerides
        # need to save off time histories of dx
        self.dx_ephem = [dx_0]
        self.P_ephem = [P_0]

    def propagate(self, y_data):
        # initialize dx and P for first prediction step

        # main propagation loop, call update and correct at each time step
        for k in range(np.shape(y_data)[1]):

            if k == 0: # no measurement at t = 0 :(
                continue

            nominal_state = np.array(self.nominal_ephem[k])
            nominal_control = self.nominal_controls[k]

            # compute needed matrices
            F_bar, G_bar = self.combined_system.get_dt_state_transition_matrices(self.dt, nominal_state, nominal_control)
            H_bar, Omega_bar = self.combined_system.get_dt_H_and_Omega(self.dt, nominal_state, nominal_control)
            self.update(F_bar, G_bar, H_bar, Omega_bar)
            # todo: index correct spot

            nominal_measurement = self.nominal_measurements[k]
            actual_measurement = y_data[:, k]
            self.correct(actual_measurement, nominal_measurement, H_bar)

            # add to ephem
            self.dx_ephem.append(self.dx_post)
            self.P_ephem.append(self.P_post)

    def update(self, F_bar, G_bar, H_bar, Omega_bar):

        # get F and G about these nominal inputs
        self.dx_pre = F_bar @ self.dx_post + G_bar @ self.du_prev

        F_bar_t = np.linalg.matrix_transpose(F_bar)
        Omega_bar_t = np.linalg.matrix_transpose(Omega_bar)
        self.P_pre = F_bar @ self.P_post @ F_bar_t + Omega_bar @ self.Q @ Omega_bar_t
        # todo: update du

    def correct(self, meas, nominal_measurement, H_bar):
        # get y_nom from nominal trajectory
        # update dy and kalman gain matrix first
        dy = meas - nominal_measurement
        H_bar_t = np.transpose(H_bar)
        self.Kk = self.P_pre @ H_bar_t @ np.linalg.inv(H_bar @ self.P_pre @ H_bar_t + self.R)

        self.dx_post = self.dx_pre + self.Kk @ (dy - H_bar @ self.dx_pre)
        # todo: correct size of I
        self.P_post = (np.eye(6) - self.Kk @ H_bar) @ self.P_pre
        # tb cont

class EKF():
    def __init__(self, dt, 
                 combined_system,
                 control,
                 Q, 
                 R,
                 P_0,
                 x_0):
        # dt linearized system information
        self.dt = dt
        self.combined_system = combined_system

        # process and measurement noise
        self.Q = Q
        self.R = R

        self.Kk = None
        # current values for estimated covariance and state disturbance
        self.x_pre = np.zeros((6,))
        self.x_post = x_0 # set post initially, since this is what pre will prop
        self.P_pre = np.zeros((6,6))
        self.P_post = P_0

        # adding control, even though control is constant.
        # these should stay at 0, and not update
        self.u = control

        # ephemerides
        # need to save off time histories of dx
        self.x_ephem = [x_0]
        self.y_hat_ephem = []
        self.P_ephem = [P_0]
        self.S_ephem = []
    

    def propagate(self, y_data):

        # main propagation loop, call update and correct at each time step
        for k in range(np.shape(y_data)[1]):

            if k == 0: # no measurement at t = 0 :(
                continue

            # compute needed matrices
            F = self.finite_difference_F(self.x_post, self.u)
            G = self.finite_difference_G(self.x_post, self.u)
            H = self.finite_difference_H(self.x_post, self.u)
            Omega = np.eye(6) / self.dt

            self.update(F, G, H, Omega)
            # todo: index correct spot

            actual_measurement = y_data[:, k]
            self.correct(actual_measurement, H)

            # add to ephem
            self.x_ephem.append(self.x_post.copy())
            self.P_ephem.append(self.P_post.copy())

            # didnt need this in LKF but we gotta update the combined systems state
            # self.combined_system.current_state = list(self.x_post)
            # still dont need it

    def update(self, F, G, H, Omega):

        # get F and G about these nominal inputs
        self.x_pre = self.combined_system.step_nl_propagation(self.u, self.dt, state=self.x_post)
        y_hat = self.combined_system.create_measurements_from_states(self.x_pre, None)
        self.y_hat_ephem.append(y_hat)
        #could be maybe?  self.combined_system.step_nl_propagation(self.u, self.dt)
        # could be maybe ? self.x_pre = np.array(self.combined_system.current_state)

        F_t = np.linalg.matrix_transpose(F)
        Omega_t = np.linalg.matrix_transpose(Omega)
        self.P_pre = F @ self.P_post @ F_t + Omega @ self.Q @ Omega_t
        # todo: update du

        self.enforce_pos_semi_def(self.P_pre, F=F, Omega=Omega)


    def correct(self, measurement, H):
        # get y_nom from nominal trajectory
        # update dy and kalman gain matrix first

        nonlinear_measurement = self.combined_system.create_measurements_from_states(state=self.x_pre)

        H_t = np.transpose(H)
        Sk = H @ self.P_pre @ H_t + self.R
        self.S_ephem.append(Sk)
        self.Kk = self.P_pre @ H_t @ np.linalg.inv(Sk)

        self.x_post = self.x_pre + self.Kk @ (measurement - nonlinear_measurement)
        # todo: correct size of I
        self.P_post = (np.eye(6) - self.Kk @ H) @ self.P_pre
        # tb cont

        self.enforce_pos_semi_def(self.P_post, H=H, correct_step=True)

    def enforce_pos_semi_def(self, mat, F = None, Omega=None, H = None, correct_step=False):

        if not (np.allclose(mat, np.transpose(mat))):
            if not (correct_step):
                print("Matrix is no longer postive semidefinite during prediction")
            else:
                print("Matrix is no longer postive semidefinite during correction")

            print(mat)
        #get the covarince's eigen values
        eigen_vals = np.linalg.eigvalsh(mat)

        if not (np.all(eigen_vals >= -1e-9)):
            if not (correct_step):
                print("Matrix is no longer postive semidefinite during prediction")

                print("F Matrix ---------------")
                print(F)
                print("Omega Matrix ---------------")
                print(Omega)
                print("Last Cov ----------")
                print(self.P_post)

            else:
                print("Matrix is no longer postive semidefinite during correction")

                print("H Matrix ---------------")
                print(H)
                print("Kalman Matrix ---------------")
                print(self.Kk)
                print("Last Cov ----------")
                print(self.P_pre)


            exit(0)

        return
            

    def finite_difference_F(self, x, u):
        '''finite difference computation of F'''
        F = np.zeros((6,6))
        epsilon = 1e-8

        x_prop_nominal = self.combined_system.step_nl_propagation(u, self.dt, state=x)

        for i in range(6):
            x_copy = x.copy() 
            x_copy[i] += epsilon

            x_prop = self.combined_system.step_nl_propagation(u, self.dt, state=x_copy)

            F[:, i] = (x_prop - x_prop_nominal) / epsilon
        
        return F
    
    def finite_difference_G(self, x, u):
        '''finite difference computation of G'''
        G = np.zeros((6,4))
        epsilon = 1e-8

        x_prop_nominal = self.combined_system.step_nl_propagation(u, self.dt, state=x)

        for i in range(4):
            u_copy = u.copy() 
            u_copy[i] += epsilon

            x_prop = self.combined_system.step_nl_propagation(u_copy, self.dt, state=x)

            G[:, i] = (x_prop - x_prop_nominal) / epsilon

        return G

    def finite_difference_H(self, x, u):
        '''finite difference computation of H'''
        H = np.zeros((5,6))
        epsilon = 1e-8

        y_nominal = self.combined_system.create_measurements_from_states(state=x)

        for i in range(6):
            x_copy = x.copy() 
            x_copy[i] += epsilon

            y_perturbed = self.combined_system.create_measurements_from_states(state=x_copy)

            H[:, i] = (y_perturbed - y_nominal) / epsilon

        return H
            