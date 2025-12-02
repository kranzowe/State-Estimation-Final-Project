''' Filter classes of project (LKF and EKF)'''

import numpy as np

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
        self.du_prev = np.zeros((4,))
        self.du = np.zeros((4,))

        # ephemerides
        # need to save off time histories of dx
        self.x_ephem = [x_0]
        self.P_ephem = [P_0]

    def propagate(self, y_data):
        # initialize dx and P for first prediction step

        # main propagation loop, call update and correct at each time step
        for k in range(np.shape(y_data)[1]):

            if k == 0: # no measurement at t = 0 :(
                continue

            #### START HERE

            # compute needed matrices
            F_bar, G_bar = self.combined_system.get_dt_state_transition_matrices(self.dt, x_, nominal_control)
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


