''' Filter classes of project (LKF and EKF)'''

import numpy as np

# todo: bar matrices need k subscript and need to be evaled at x_star, u_star at each step before update and correct
class LKF():
    def __init__(self, dt, F_bar, G_bar, Omega_bar, H_bar, Q, R):
        # dt linearized system information
        self.dt = dt
        self.F_bar = F_bar
        self.G_bar = G_bar
        self.Omega_bar = Omega_bar
        self.H_bar = H_bar

        # process and measurement noise
        self.Q = Q
        self.R = R

        # need to save off time histories of dx, dy

        # kalman gain matrix initialized
        self.Kk = np.zeros(1, 1)
        # current values for estimated covariance and state disturbance
        self.dx_init = 0
        self.dx_pre = 0
        self.dx_post = 0
        self.P_init = 0
        self.P_pre = 0
        self.P_post = 0

        self.du_prev = 0
        self.du = 0

    def propagate(self, P_0, dx_0, t_vec, y_data):
        # initialize dx and P for first prediction step
        self.dx_post = self.dx_init
        self.P_post = self.P_init

        # main propagation loop, call update and correct at each time step
        for k in enumerate(t_vec):
            if k==0:
                continue
            self.update()
            # todo: index correct spot
            meas = y_data(k)
            self.correct(meas)

    def update(self):
        self.dx_pre = self.F_bar @ self.dx_post + self.G_bar @ self.du_prev

        F_bar_t = np.linalg.matrix_transpose(self.F_bar)
        Omega_bar_t = np.linalg.matrix_transpose(self.Omega_bar)
        self.P_pre = self.F_bar @ self.P_post @ F_bar_t + self.Omega_bar @ self.Q @ Omega_bar_t
        # todo: update du

    def correct(self, meas):
        # get y_nom from nominal trajectory
        y_nom = 0
        # update dy and kalman gain matrix first
        dy = meas - y_nom
        H_bar_t = np.transpose(self.H_bar)
        self.Kk = self.P_pre @ H_bar_t @ np.linalg.inv(self.H_bar @ self.P_pre @ H_bar_t + self.R)

        self.dx_post = self.dx_pre + self.Kk @ (dy - self.H_bar @ self.dx_pre)
        # todo: correct size of I
        self.P_post = (np.eye(1) - self.Kk @ self.H_bar) @ self.P_pre
        # tb cont

# class EKF():
#     def __init__(self):
#         self.dt = 0.05
#
#     def propagate(self):
#
#     def update(self):
#
#     def correct(self):


