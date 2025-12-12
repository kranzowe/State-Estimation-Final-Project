'''tests the linearized kalman filter'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import src.ugv_dynamics as ugv_dynamics
import src.uav_dynamics as uav_dynamics
import src.combined_system as combined_system
from src import filters
import math

from scipy.stats import chi2

import matplotlib.pyplot as plt

NUM_TESTS = 1
NUM_TESTING_STEPS = 1000
SIGNFICANCE_LEVEL = 0.01

# def test_lkf():
#     """gen the plots as shown in the pdf"""
#     # assumed nominal trajectory
#     x0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
#     # x0 += np.array([0, 1, 0, 0, 0, 0.1])

#     # constant control
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])

#     ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)

#     dt = 0.1
#     t = 0


#     # need to generate a nominal ephem and control
#     nominal_ephemeris = [x0]
#     nominal_controls = [control]
#     times = [t]
#     nominal_measurements = [combo.create_measurements_from_states()]
#     while t <= 100-dt:
#         combo.step_nl_propagation(control, dt)
#         nominal_ephemeris.append(combo.current_state)
#         nominal_controls.append(control)
#         nominal_measurements.append(combo.create_measurements_from_states())
#         t += dt
#         times.append(t)

#     #
#     x0 += np.array([0, 1, 0, 0, 0, 0.1])
#     R_true = np.array([[0.0225,0,0,0,0],
#                         [0,64,0,0,0],
#                         [0,0,0.04,0,0],
#                         [0,0,0,36,0],
#                         [0,0,0,0,36]])
#     Q_true = np.array([[0.001,0,0,0,0,0],
#                         [0,0.001,0,0,0,0],
#                         [0,0,0.01,0,0,0],
#                         [0,0,0,0.001,0,0],
#                         [0,0,0,0,0.001,0],
#                         [0,0,0,0,0,0.01]])

#     P_0 = np.eye(6) * 10
#     dx_0 = np.zeros((6,))

#     # now run the filter
#     # reset class
#     ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)
#     lkf = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
#                       combo,
#                       Q_true,
#                       R_true,
#                       P_0,
#                       dx_0)


#     # y_data = np.loadtxt(f"../src/data/ydata.csv", delimiter=",")
#     # t_vec = np.loadtxt(f"../src/data/tvec.csv", delimiter=",")

#     # todo: temp usage of truth model
#     ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)
#     tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2],
#                                                                       control[2:4], True)
#     y_data = tmt_measurement



#     lkf.propagate(y_data)

#     # plot ephem  

#     # ai helped me plot cuz eww
#     nominal_ephemeris = np.array(nominal_ephemeris)
#     tmt_states = np.concatenate((x0.reshape(1, 6), tmt_states), axis=0)
#     true_dx = tmt_states - nominal_ephemeris
#     # true_dx = np.array([0, 1, 0, 0, 0, 0.1])
#     ephemeris = np.array(lkf.dx_ephem)
#     P_history = np.array(lkf.P_ephem)
    
#     # Extract 2-sigma bounds for each state (skip first timestep)
#     sigma_bounds = np.zeros((len(P_history), 6))
#     for i in range(len(P_history)):
#         sigma_bounds[i, :] = 2 * np.sqrt(np.diag(P_history[i]))
    
#     # Create figure with 6 subplots
#     fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    
#     # Plot zeta_g
#     axes[0].plot(times[:-1], ephemeris[:, 0], label='Estimated')
#     axes[0].fill_between(times[:-1], ephemeris[:, 0] - sigma_bounds[:, 0], 
#                          ephemeris[:, 0] + sigma_bounds[:, 0], alpha=0.3, label='2σ bounds')
#     axes[0].plot(times, true_dx[:, 0], color='r', linestyle='--', label='True')
#     axes[0].set_ylabel('dζ_g (m)')
#     axes[0].legend()
#     axes[0].grid(True)
    
#     # Plot eta_g
#     axes[1].plot(times[:-1], ephemeris[:, 1], label='Estimated')
#     axes[1].fill_between(times[:-1], ephemeris[:, 1] - sigma_bounds[:, 1], 
#                          ephemeris[:, 1] + sigma_bounds[:, 1], alpha=0.3, label='2σ bounds')
#     axes[1].plot(times, true_dx[:, 1], color='r', linestyle='--', label='True')
#     axes[1].set_ylabel('dη_g (m)')
#     axes[1].legend()
#     axes[1].grid(True)
    
#     # Plot theta_g
#     axes[2].plot(times[:-1], ephemeris[:, 2], label='Estimated')
#     axes[2].fill_between(times[:-1], ephemeris[:, 2] - sigma_bounds[:, 2], 
#                          ephemeris[:, 2] + sigma_bounds[:, 2], alpha=0.3, label='2σ bounds')
#     axes[2].plot(times, true_dx[:, 2], color='r', linestyle='--', label='True')
#     axes[2].set_ylabel('dθ_g (rad)')
#     axes[2].legend()
#     axes[2].grid(True)

#     # Plot zeta_a
#     axes[3].plot(times[:-1], ephemeris[:, 3], label='Estimated')
#     axes[3].fill_between(times[:-1], ephemeris[:, 3] - sigma_bounds[:, 3], 
#                          ephemeris[:, 3] + sigma_bounds[:, 3], alpha=0.3, label='2σ bounds')
#     axes[3].plot(times, true_dx[:, 3], color='r', linestyle='--', label='True')
#     axes[3].set_ylabel('dζ_a (m)')
#     axes[3].legend()
#     axes[3].grid(True)
    
#     # Plot eta_a
#     axes[4].plot(times[:-1], ephemeris[:, 4], label='Estimated')
#     axes[4].fill_between(times[:-1], ephemeris[:, 4] - sigma_bounds[:, 4], 
#                          ephemeris[:, 4] + sigma_bounds[:, 4], alpha=0.3, label='2σ bounds')
#     axes[4].plot(times, true_dx[:, 4], color='r', linestyle='--', label='True')
#     axes[4].set_ylabel('dη_a (m)')
#     axes[4].legend()
#     axes[4].grid(True)
    
#     # Plot theta_a
#     axes[5].plot(times[:-1], ephemeris[:, 5], label='Estimated')
#     axes[5].fill_between(times[:-1], ephemeris[:, 5] - sigma_bounds[:, 5], 
#                          ephemeris[:, 5] + sigma_bounds[:, 5], alpha=0.3, label='2σ bounds')
#     axes[5].plot(times, true_dx[:, 5], color='r', linestyle='--', label='True')
#     axes[5].set_xlabel('Time (s)')
#     axes[5].set_ylabel('dθ_a (rad)')
#     axes[5].legend()
#     axes[5].grid(True)
    
#     plt.tight_layout()
#     plt.show()

# def test_lkf_nees():
#     np.random.seed(0)
#     """gen the plots as shown in the pdf"""
#     # assumed nominal trajectory
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])

#     # constant control
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])

#     ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)

#     dt = 0.1
#     t = 0

#     dx_0 = np.zeros((6,))


#     # ## generate truth ephem
#     # truth_ephemeris = [x_0]
#     # times = [t]
#     # while t <=100:
#     #     combo.step_nl_propagation(control, dt)
#     #     truth_ephemeris.append(combo.current_state)
#     #     t += dt
#     #     times.append(t)

#     # # ai helped me plot cuz eww
#     # truth_ephemeris = np.array(truth_ephemeris)

#     #
#     nominal_ephemeris = [x_0]
#     nominal_controls = [control]
#     times = [t]
#     nominal_measurements = [combo.create_measurements_from_states()]
#     while t < NUM_TESTING_STEPS * dt:
#         combo.step_nl_propagation(control, dt)
#         nominal_ephemeris.append(combo.current_state)
#         nominal_controls.append(control)
#         nominal_measurements.append(combo.create_measurements_from_states())
#         t += dt
#         times.append(t)

#     # now apply perturbation
#     x_0 += np.array([0, 1, 0, 0, 0, 0.1])

#     R_true = np.array([[0.0225,0,0,0,0],
#                         [0,64,0,0,0],
#                         [0,0,0.04,0,0],
#                         [0,0,0,36,0],
#                         [0,0,0,0,36]])
#     Q_true = np.array([[0.001,0,0,0,0,0],
#                         [0,0.001,0,0,0,0],
#                         [0,0,0.01,0,0,0],
#                         [0,0,0,0.001,0,0],
#                         [0,0,0,0,0.001,0],
#                         [0,0,0,0,0,0.01]])


#     P_0 = np.eye(6) * 10

#     nees_sum = np.zeros([NUM_TESTING_STEPS, 6])

#     for _ in range(0, NUM_TESTS):

#         ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#         uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#         combo = combined_system.CombinedSystem(ugv, uav)

#         lkf: filters.LKF = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
#                       combo,
#                       Q_true,
#                       R_true,
#                       P_0,
#                       dx_0)  

#         #generate the truth model to run the nees testing on
#         tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4])

#         # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
#         # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

#         lkf.propagate(tmt_measurement)

#         x_ephem = np.array(lkf.dx_ephem) + np.array(nominal_ephemeris)[1:NUM_TESTING_STEPS+1]

#         for step, cov in enumerate(lkf.P_ephem):
#             diff = x_ephem[step,:] - tmt_states[step,:]
#             diff[2] = filters.wrap_angle(diff[2])
#             diff[5] = filters.wrap_angle(diff[5])
#             nees_sum[step,:] += diff @ np.linalg.inv(cov) @ np.transpose(diff)
#         # todo: temp debugging plot
#         # Create figure with 6 subplots
#         fig, axes = plt.subplots(6, 1, figsize=(10, 12))

#         # Plot zeta_g
#         axes[0].plot(tmt_times, x_ephem[:, 0], label='Estimated')
#         axes[0].plot(tmt_times, tmt_states[:, 0], linestyle='--', label='true')
#         axes[0].set_ylabel('dζ_g (m)')
#         axes[0].legend()
#         axes[0].grid(True)

#         # Plot eta_g
#         axes[1].plot(tmt_times, x_ephem[:, 1], label='Estimated')
#         axes[1].plot(tmt_times, tmt_states[:, 1], linestyle='--', label='true')
#         axes[1].set_ylabel('dη_g (m)')
#         axes[1].legend()
#         axes[1].grid(True)

#         # Plot theta_g
#         axes[2].plot(tmt_times, x_ephem[:, 2], label='Estimated')
#         axes[2].plot(tmt_times, tmt_states[:, 2], linestyle='--', label='true')
#         axes[2].set_ylabel('dθ_g (rad)')
#         axes[2].legend()
#         axes[2].grid(True)

#         # Plot zeta_a
#         axes[3].plot(tmt_times, x_ephem[:, 3], label='Estimated')
#         axes[3].plot(tmt_times, tmt_states[:, 3], linestyle='--', label='true')
#         axes[3].set_ylabel('dζ_a (m)')
#         axes[3].legend()
#         axes[3].grid(True)

#         # Plot eta_a
#         axes[4].plot(tmt_times, x_ephem[:, 4], label='Estimated')
#         axes[4].plot(tmt_times, tmt_states[:, 4], linestyle='--', label='true')
#         axes[4].set_ylabel('dη_a (m)')
#         axes[4].legend()
#         axes[4].grid(True)

#         # Plot theta_a
#         axes[5].plot(tmt_times, x_ephem[:, 5], label='Estimated')
#         axes[5].plot(tmt_times, tmt_states[:, 5], linestyle='--', label='true')
#         axes[5].set_xlabel('Time (s)')
#         axes[5].set_ylabel('dθ_a (rad)')
#         axes[5].legend()
#         axes[5].grid(True)


#     nees_sum = nees_sum / NUM_TESTS

#     # #determine the chi2inv for upper and lower error bound
#     r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) / NUM_TESTS
#     r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) /  NUM_TESTS
   
#     fig, axes = plt.subplots(1, 1, figsize=(10, 12))

#     axes.plot(tmt_times, nees_sum[:, 0], marker='o', linestyle="none", color='blue')
#     axes.plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
#     axes.plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')
#     axes.grid(True)


#     plt.tight_layout()
#     plt.show()



# def test_lkf_nis():
#     np.random.seed(0)
#     """gen the plots as shown in the pdf"""
#     # assumed nominal trajectory
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])

#     # constant control
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])

#     ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)

#     dt = 0.1
#     t = 0

#     dx_0 = np.zeros((6,))

#     # get nominal trajectory
#     nominal_ephemeris = [x_0]
#     nominal_controls = [control]
#     times = [t]
#     nominal_measurements = [combo.create_measurements_from_states()]
#     while t < NUM_TESTING_STEPS * dt:
#         combo.step_nl_propagation(control, dt)
#         nominal_ephemeris.append(combo.current_state)
#         nominal_controls.append(control)
#         nominal_measurements.append(combo.create_measurements_from_states())
#         t += dt
#         times.append(t)

#     # now apply perturbation
#     x_0 += np.array([0, 1, 0, 0, 0, 0.1])

#     R_true = np.array([[0.0225,0,0,0,0],
#                         [0,64,0,0,0],
#                         [0,0,0.04,0,0],
#                         [0,0,0,36,0],
#                         [0,0,0,0,36]])
#     Q_true = np.array([[0.001,0,0,0,0,0],
#                         [0,0.001,0,0,0,0],
#                         [0,0,0.01,0,0,0],
#                         [0,0,0,0.001,0,0],
#                         [0,0,0,0,0.001,0],
#                         [0,0,0,0,0,0.01]])


#     P_0 = np.eye(6) * 10

#     nis_sum = np.zeros([NUM_TESTING_STEPS, 6])
#     # todo temp
#     tmt_times = []
#     y_ephem = []
#     tmt_measurement = []
#     for _ in range(0, NUM_TESTS):

#         ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#         uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#         combo = combined_system.CombinedSystem(ugv, uav)

#         lkf: filters.LKF = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
#                                        combo,
#                                        Q_true,
#                                        R_true,
#                                        P_0,
#                                        dx_0)

#         # generate the truth model to run the nees testing on
#         tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2],
#                                                                           control[2:4])

#         # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
#         # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

#         lkf.propagate(tmt_measurement)

#         y_ephem = np.array(lkf.dy_ephem) + np.array(nominal_measurements)[1:NUM_TESTING_STEPS]
#         # x_ephem = np.array(lkf.dx_ephem) + np.array(nominal_ephemeris)[1:NUM_TESTING_STEPS + 1]
#         H_ephem = np.array(lkf.H_ephem)
#         P_pre_ephem = np.array(lkf.P_pre_ephem)

#         for step, (P_pre, H) in enumerate(zip(P_pre_ephem, H_ephem)):
#             diff = y_ephem[step,:] - np.transpose(tmt_measurement[:, step+1])
#             diff[0] = filters.wrap_angle(diff[0])
#             diff[2] = filters.wrap_angle(diff[2])

#             S = H @ P_pre @ np.transpose(H) + lkf.R
#             nis_sum[step, :] += (diff) @ np.linalg.inv(S) @ np.transpose(diff)

#     nis_sum = nis_sum / NUM_TESTS

#     # #determine the chi2inv for upper and lower error bound
#     r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS
#     r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS

#     # todo: temp debugging plot
#     # Create figure with 6 subplots
#     fig, axes = plt.subplots(5, 1, figsize=(10, 12))

#     # Plot zeta_g
#     axes[0].plot(tmt_times[1:], y_ephem[:, 0], label='Estimated')
#     axes[0].plot(tmt_times, tmt_measurement[0, :], linestyle='--', label='true')
#     axes[0].set_ylabel('dζ_g (m)')
#     axes[0].legend()
#     axes[0].grid(True)

#     # Plot eta_g
#     axes[1].plot(tmt_times[1:], y_ephem[:, 1], label='Estimated')
#     axes[1].plot(tmt_times, tmt_measurement[1, :], linestyle='--', label='true')
#     axes[1].set_ylabel('dη_g (m)')
#     axes[1].legend()
#     axes[1].grid(True)

#     # Plot theta_g
#     axes[2].plot(tmt_times[1:], y_ephem[:, 2], label='Estimated')
#     axes[2].plot(tmt_times, tmt_measurement[2, :], linestyle='--', label='true')
#     axes[2].set_ylabel('dθ_g (rad)')
#     axes[2].legend()
#     axes[2].grid(True)

#     # Plot zeta_a
#     axes[3].plot(tmt_times[1:], y_ephem[:, 3], label='Estimated')
#     axes[3].plot(tmt_times, tmt_measurement[3, :], linestyle='--', label='true')
#     axes[3].set_ylabel('dζ_a (m)')
#     axes[3].legend()
#     axes[3].grid(True)

#     # Plot eta_a
#     axes[4].plot(tmt_times[1:], y_ephem[:, 4], label='Estimated')
#     axes[4].plot(tmt_times, tmt_measurement[4, :], linestyle='--', label='true')
#     axes[4].set_ylabel('dη_a (m)')
#     axes[4].legend()
#     axes[4].grid(True)

#     fig, axes = plt.subplots(1, 1, figsize=(10, 12))

#     axes.plot(tmt_times, nis_sum[:, 0], marker='o', linestyle="none", color='blue')
#     axes.plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
#     axes.plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')
#     axes.grid(True)

#     plt.tight_layout()
#     plt.show()

    

def test_lkf_comprehensive():
    """Comprehensive LKF test: NEES, NIS, state residuals, and measurement residuals"""
    
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    control = np.array([2, -math.pi / 18, 12, math.pi/25])
    dt = 0.1
    
    R_true = np.array([[0.0225,0,0,0,0],
                        [0,64,0,0,0],
                        [0,0,0.04,0,0],
                        [0,0,0,36,0],
                        [0,0,0,0,36]])
    Q_true = np.array([[0.001,0,0,0,0,0],
                        [0,0.001,0,0,0,0],
                        [0,0,0.01,0,0,0],
                        [0,0,0,0.001,0,0],
                        [0,0,0,0,0.001,0],
                        [0,0,0,0,0,0.01]]) /6
    
    P_0 = np.diag([1.0, 1.0, 0.01, 1.0, 1.0, 0.01])
    dx_0 = np.zeros((6,))

    #record single test for report
    single_measurements = None
    single_times = None
    single_states= None
    single_error = []
    single_two_sigma = []

    
    # Generate nominal trajectory
    ugv_nom = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav_nom = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo_nom = combined_system.CombinedSystem(ugv_nom, uav_nom)
    
    t = 0
    nominal_ephemeris = [x_0]
    nominal_controls = [control]
    nominal_measurements = [combo_nom.create_measurements_from_states()]
    while t <= NUM_TESTING_STEPS * dt:
        combo_nom.step_nl_propagation(control, dt)
        nominal_ephemeris.append(combo_nom.current_state)
        nominal_controls.append(control)
        nominal_measurements.append(combo_nom.create_measurements_from_states())
        t += dt
    
    # NEES and NIS Testing
    nees_sum = np.zeros([NUM_TESTING_STEPS, 1])
    nis_sum = np.zeros([NUM_TESTING_STEPS - 1, 1])  # missing one measurement
    
    for test_num in range(NUM_TESTS):
        # Sample initial state with uncertainty
        x_0_truth = x_0 + np.random.multivariate_normal(np.zeros(6), P_0)
        
        ugv = ugv_dynamics.Dynamical_UGV(x_0_truth[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0_truth[3:])
        combo = combined_system.CombinedSystem(ugv, uav)
        
        # Generate truth with process noise
        tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(
            dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4], process_noise=True
        )

        if(single_measurements is None):
            single_times = tmt_times
            single_measurements = tmt_measurement
            single_states = tmt_states
        
        # Run LKF from nominal trajectory with initial perturbation
        ugv_lkf = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav_lkf = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo_lkf = combined_system.CombinedSystem(ugv_lkf, uav_lkf)
        
        dx_0_sample = x_0_truth - x_0  # Initial perturbation
        lkf = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
                         combo_lkf, Q_true, R_true, P_0, dx_0_sample)
        
        lkf.propagate(tmt_measurement)
        
        dx_ephem = np.array(lkf.dx_ephem)
        dy_ephem = np.array(lkf.dy_ephem)
        
        # Reconstruct full state: x = x_nom + dx
        x_ephem = np.array([nominal_ephemeris[i] + dx_ephem[i] for i in range(len(dx_ephem))])
        
        # Compute NEES
        for step, cov in enumerate(lkf.P_ephem):
            if step == 0:
                continue

            if(test_num == 0):
                single_error.append(x_ephem[step,:] - tmt_states[step - 1,:])
                single_two_sigma.append( [
                    math.sqrt(cov[0,0]) * 2,
                    math.sqrt(cov[1,1]) * 2,
                    math.sqrt(cov[2,2]) * 2,
                    math.sqrt(cov[3,3]) * 2,
                    math.sqrt(cov[4,4]) * 2,
                    math.sqrt(cov[5,5]) * 2])
                
            if(single_error[-1][2] > math.pi):
                single_error[-1][2] -= math.pi * 2
            elif(single_error[-1][2] < -math.pi):
                single_error[-1][2] += math.pi * 2
            if(single_error[-1][5] > math.pi):
                single_error[-1][5] -= math.pi * 2
            elif(single_error[-1][5] < -math.pi):
                single_error[-1][5] += math.pi * 2

            
            state_error = x_ephem[step] - tmt_states[step - 1]
            state_error[2] = filters.angle_difference(x_ephem[step, 2], tmt_states[step - 1, 2])
            state_error[5] = filters.angle_difference(x_ephem[step, 5], tmt_states[step - 1, 5])
            
            nees_sum[step - 1, :] += state_error @ np.linalg.inv(cov) @ state_error
        
        # Compute NIS
        for step in range(1, len(dy_ephem)):
            y_pred = nominal_measurements[step] + dy_ephem[step - 1]
            y_actual = tmt_measurement[:, step]  # Changed from step-1 to step
            
            measurement_error = y_actual - y_pred
            measurement_error[0] = filters.angle_difference(y_actual[0], y_pred[0])
            measurement_error[2] = filters.angle_difference(y_actual[2], y_pred[2])
            
            innovation_cov = lkf.H_ephem[step - 1] @ lkf.P_pre_ephem[step - 1] @ lkf.H_ephem[step - 1].T + R_true
            nis_sum[step - 1, :] += measurement_error @ np.linalg.inv(innovation_cov) @ measurement_error
    
    nees_sum = nees_sum / NUM_TESTS
    nis_sum = nis_sum / NUM_TESTS
    
    # Chi-squared bounds
    r1_nees = chi2.ppf(SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) / NUM_TESTS
    r2_nees = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) / NUM_TESTS
    r1_nis = chi2.ppf(SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS
    r2_nis = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS
    
    # ========================================================================
    # PART 2: State Residuals (vs truth with no process noise)
    # ========================================================================
    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo_truth = combined_system.CombinedSystem(ugv, uav)
    
    # Generate truth WITHOUT process noise
    truth_times, truth_states, truth_measurements = combo_truth.generate_truth_set(
        dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4], process_noise=False
    )
    
    # Run filter on this data
    ugv_state = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav_state = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo_state = combined_system.CombinedSystem(ugv_state, uav_state)
    
    lkf_state = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
                           combo_state, Q_true, R_true, P_0, dx_0)
    
    lkf_state.propagate(truth_measurements)
    
    dx_ephem_state = np.array(lkf_state.dx_ephem)
    x_ephem_state = np.array([nominal_ephemeris[i] + dx_ephem_state[i] for i in range(len(dx_ephem_state))])
    
    # Compute state residuals - fix range to match EKF
    state_residuals = np.zeros((NUM_TESTING_STEPS-1, 6))
    for i in range(NUM_TESTING_STEPS-1):
        state_residuals[i, :] = x_ephem_state[i + 1, :] - truth_states[i, :]
        state_residuals[i, 2] = filters.angle_difference(x_ephem_state[i + 1, 2], truth_states[i, 2])
        state_residuals[i, 5] = filters.angle_difference(x_ephem_state[i + 1, 5], truth_states[i, 5])

    # ========================================================================
    # PART 3: Measurement Residuals (vs ydata.csv)
    # ========================================================================
    y_data = np.loadtxt(f"{os.getcwd()}/src/data/ydata.csv", delimiter=",")
    t_vec = np.loadtxt(f"{os.getcwd()}/src/data/tvec.csv", delimiter=",")
    
    if y_data.shape[0] != 5:
        y_data = y_data.T
    
    # Skip first measurement (NaN at t=0)
    y_data = y_data[:, 1:]
    t_vec_data = t_vec[1:]
    
    # Generate nominal trajectory for ydata length
    ugv_nom2 = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav_nom2 = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo_nom2 = combined_system.CombinedSystem(ugv_nom2, uav_nom2)
    
    t = 0
    nominal_ephemeris2 = [x_0]
    nominal_controls2 = [control]
    nominal_measurements2 = [combo_nom2.create_measurements_from_states()]
    while t < len(t_vec_data) * dt:
        combo_nom2.step_nl_propagation(control, dt)
        nominal_ephemeris2.append(combo_nom2.current_state)
        nominal_controls2.append(control)
        nominal_measurements2.append(combo_nom2.create_measurements_from_states())
        t += dt
    
    # Run filter on ydata
    x_0_data = x_0 + np.array([0, 1, 0, 0, 0, 0.1])
    dx_0_data = np.array([0, 1, 0, 0, 0, 0.1])
    
    ugv_meas = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav_meas = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo_meas = combined_system.CombinedSystem(ugv_meas, uav_meas)
    
    lkf_meas = filters.LKF(dt, nominal_ephemeris2, nominal_controls2, nominal_measurements2,
                          combo_meas, Q_true, R_true, P_0, dx_0_data)
    
    lkf_meas.propagate(y_data)
    

    real_data_states = np.array(lkf.dx_ephem) + np.array(nominal_ephemeris2[:-2])
    real_data_states_sigma = []
    for cov in lkf.P_ephem:
        real_data_states_sigma.append([
            math.sqrt(cov[0,0]) * 2,
            math.sqrt(cov[1,1]) * 2,
            math.sqrt(cov[2,2]) * 2,
            math.sqrt(cov[3,3]) * 2,
            math.sqrt(cov[4,4]) * 2,
            math.sqrt(cov[5,5]) * 2,
        ])
    real_data_states_sigma = np.array(real_data_states_sigma)

    dy_ephem_data = np.array(lkf_meas.dy_ephem)
    
    # Reconstruct predicted measurements
    y_pred_data = np.array([nominal_measurements2[i + 1] + dy_ephem_data[i] for i in range(len(dy_ephem_data))])
    
    # Compute measurement residuals
    meas_residuals = np.zeros((len(t_vec_data), 5))
    for i in range(min(len(t_vec_data), len(y_pred_data))):
        meas_residuals[i, :] = y_data[:, i] - y_pred_data[i]
        meas_residuals[i, 0] = filters.angle_difference(y_data[0, i], y_pred_data[i, 0])
        meas_residuals[i, 2] = filters.angle_difference(y_data[2, i], y_pred_data[i, 2])

    single_error = np.array(single_error)
    single_two_sigma = np.array(single_two_sigma)
    
    # ========================================================================
    # PLOTTING
    # ========================================================================
    
    # Plot 1: NEES
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(tmt_times, nees_sum[:, 0], marker='o', linestyle='none', color='blue', label='NEES')
    ax1.axhline(r1_nees, linestyle='--', color='red', label=f'Lower bound ({r1_nees:.2f})')
    ax1.axhline(r2_nees, linestyle='--', color='red', label=f'Upper bound ({r2_nees:.2f})')
    ax1.axhline(6.0, linestyle=':', color='green', label='Expected (6.0)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('NEES')
    ax1.set_title('LKF - Normalized Estimation Error Squared (NEES)')
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    
    # Plot 2: NIS
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(tmt_times[1:], nis_sum[:, 0], marker='o', linestyle='none', color='orange', label='NIS')
    ax2.axhline(r1_nis, linestyle='--', color='red', label=f'Lower bound ({r1_nis:.2f})')
    ax2.axhline(r2_nis, linestyle='--', color='red', label=f'Upper bound ({r2_nis:.2f})')
    ax2.axhline(5.0, linestyle=':', color='green', label='Expected (5.0)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('NIS')
    ax2.set_title('LKF - Normalized Innovation Squared (NIS)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    
    # Plot 3: State Residuals
    fig3, axes3 = plt.subplots(6, 1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    
    for i in range(6):
        axes3[i].plot(truth_times[:-1], state_residuals[:, i], 'b-', alpha=0.6, linewidth=1)
        axes3[i].axhline(0, color='k', linestyle='--', linewidth=1)
        mean_res = np.mean(state_residuals[:, i])
        std_res = np.std(state_residuals[:, i])
        axes3[i].axhline(mean_res, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_res:.4f}')
        axes3[i].fill_between(truth_times, -std_res, std_res, alpha=0.2, color='blue', label=f'±1σ = {std_res:.4f}')
        axes3[i].set_ylabel(state_labels[i])
        axes3[i].set_xlabel('Time (s)')
        axes3[i].legend(loc='upper right')
        axes3[i].grid(True, alpha=0.3)
        axes3[i].set_title(f'LKF State Residual: {state_labels[i]} (Estimated - Truth)')
    
    plt.tight_layout()
    
    # Plot 4: Measurement Residuals
    fig4, axes4 = plt.subplots(5, 1, figsize=(12, 14))
    meas_labels = ['Bearing UGV→UAV (rad)', 'Range (m)', 'Bearing UAV→UGV (rad)', 'UAV ζ (m)', 'UAV η (m)']
    
    for i in range(5):
        axes4[i].plot(t_vec_data[:len(y_pred_data)], meas_residuals[:len(y_pred_data), i], 'b-', alpha=0.6, linewidth=1)
        axes4[i].axhline(0, color='k', linestyle='--', linewidth=1)
        mean_res = np.mean(meas_residuals[:len(y_pred_data), i])
        std_res = np.std(meas_residuals[:len(y_pred_data), i])
        axes4[i].axhline(mean_res, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_res:.4f}')
        axes4[i].fill_between(t_vec_data[:len(y_pred_data)], -std_res, std_res, alpha=0.2, color='blue', label=f'±1σ = {std_res:.4f}')
        axes4[i].set_ylabel(meas_labels[i])
        axes4[i].set_xlabel('Time (s)')
        axes4[i].legend(loc='upper right')
        axes4[i].grid(True, alpha=0.3)
        axes4[i].set_title(f'LKF Measurement Residual: {meas_labels[i]} (Actual - Predicted)')

    plt.tight_layout()

    fig5, axes5 = plt.subplots(5,1, figsize=(12, 14))
    meas_labels = ['Bearing UGV→UAV (rad)', 'Range (m)', 'Bearing UAV→UGV (rad)', 'UAV ζ (m)', 'UAV η (m)']
    for i in range(5):
        axes5[i].plot(single_times, single_measurements[i,:], "b-")
        axes5[i].set_ylabel(meas_labels[i])

    axes5[4].set_xlabel('Time (s)')
    axes5[0].set_title(f'Truth Model Noisy Measurements for LKF NEES / NIST Testing')

    fig6, axes6 = plt.subplots(6,1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    for i in range(6):
        axes6[i].plot(single_times, single_states[:,i], "b-")
        axes6[i].set_ylabel(state_labels[i])

    axes6[5].set_xlabel('Time (s)')
    axes6[0].set_title(f'Truth Model States for LKF NEES / NIS Testing ')

    fig7, axes7 = plt.subplots(6,1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    for i in range(6):
        axes7[i].plot(single_times[:-1], single_error[:,i], "r-", label="State Error")
        axes7[i].fill_between(single_times[:-1], single_error[:,i] - single_two_sigma[:,i], single_error[:,i] + single_two_sigma[:,i],  alpha=0.2, color='blue', label=f'±2σ')
        axes7[i].set_ylabel(state_labels[i])
        axes7[i].legend(loc="upper right")

    axes7[5].set_xlabel('Time (s)')
    axes7[0].set_title(f'Truth Model States for LKF NEES / NIS Testing ')

    fig8, axes8 = plt.subplots(6,1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    for i in range(6):
        axes8[i].plot(t_vec_data, real_data_states[:,i], "r-", label="State Error")
        axes8[i].fill_between(t_vec_data, real_data_states[:,i] - real_data_states_sigma[:,i], real_data_states[:,i] + real_data_states_sigma[:,i],  alpha=0.2, color='blue', label=f'±2σ')
        axes8[i].set_ylabel(state_labels[i])
        axes8[i].legend(loc="upper right")

    axes8[5].set_xlabel('Time (s)')
    axes8[0].set_title(f'LKF State Estimate on Observation Data Log')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_lkf_comprehensive()
    # R_estimated, R_given = test_estimate_R_from_data()
    # test_q_continuous_vs_discrete()