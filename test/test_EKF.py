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

import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

NUM_TESTS = 3
NUM_TESTING_STEPS = 100
SIGNFICANCE_LEVEL = 0.01


# def test_ekf():
#     """gen the plots as shown in the pdf"""
#     # assumed nominal trajectory
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
#     x_0 += np.array([0, 1, 0, 0, 0, 0.1])

#     # constant control
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])

#     ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)

#     dt = 0.1
#     t = 0

#     ## generate truth ephem
#     truth_ephemeris = [x_0]
#     times = [t]
#     while t <=100:
#         combo.step_nl_propagation(control, dt)
#         truth_ephemeris.append(combo.current_state)
#         t += dt
#         times.append(t)

#     # ai helped me plot cuz eww
#     truth_ephemeris = np.array(truth_ephemeris)

#     #
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

#     # now run the filter
#     ekf = filters.EKF(dt,
#                       combo,
#                       control,
#                       Q_true,
#                       R_true,
#                       P_0,
#                       x_0)  

#     y_data = np.loadtxt(f"{os.getcwd()}/src/data/ydata.csv", delimiter=",")
#     t_vec = np.loadtxt(f"{os.getcwd()}/src/data/tvec.csv", delimiter=",")

#     ekf.propagate(y_data)

#     # plot ephem  

#     # ai helped me plot cuz eww
#     ephemeris = np.array(ekf.x_ephem)
#     P_history = np.array(ekf.P_ephem)
    
#     # Extract 2-sigma bounds for each state
#     sigma_bounds = np.zeros((len(P_history), 6))
#     for i in range(len(P_history)):
#         sigma_bounds[i, :] = 2 * np.sqrt(np.diag(P_history[i]))
    
#     # Create figure with 6 subplots
#     fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    
#     # Plot zeta_g
#     axes[0].plot(t_vec, ephemeris[:, 0], label='Estimated', color='blue')
#     axes[0].fill_between(t_vec, ephemeris[:, 0] - sigma_bounds[:, 0], 
#                          ephemeris[:, 0] + sigma_bounds[:, 0], alpha=0.3, color='blue', label='2σ bounds')
#     axes[0].plot(times, truth_ephemeris[:, 0], color='r', linestyle='--', label='True')
#     axes[0].set_ylabel('ζ_g (m)')
#     axes[0].set_ylim([-100, 300])
#     axes[0].legend()
#     axes[0].grid(True)
    
#     # Plot eta_g
#     axes[1].plot(t_vec, ephemeris[:, 1], label='Estimated', color='blue')
#     axes[1].fill_between(t_vec, ephemeris[:, 1] - sigma_bounds[:, 1], 
#                          ephemeris[:, 1] + sigma_bounds[:, 1], alpha=0.3, color='blue', label='2σ bounds')
#     axes[1].plot(times, truth_ephemeris[:, 1], color='r', linestyle='--', label='True')
#     axes[1].set_ylabel('η_g (m)')
#     axes[1].set_ylim([-50, 50])
#     axes[1].legend()
#     axes[1].grid(True)
    
#     # Plot theta_g
#     axes[2].plot(t_vec, ephemeris[:, 2], label='Estimated', color='blue')
#     axes[2].fill_between(t_vec, ephemeris[:, 2] - sigma_bounds[:, 2], 
#                          ephemeris[:, 2] + sigma_bounds[:, 2], alpha=0.3, color='blue', label='2σ bounds')
#     axes[2].plot(times, truth_ephemeris[:, 2], color='r', linestyle='--', label='True')
#     axes[2].set_ylabel('θ_g (rad)')
#     axes[2].set_ylim([-4, 4])
#     axes[2].legend()
#     axes[2].grid(True)

#     # Plot zeta_a
#     axes[3].plot(t_vec, ephemeris[:, 3], label='Estimated', color='blue')
#     axes[3].fill_between(t_vec, ephemeris[:, 3] - sigma_bounds[:, 3], 
#                          ephemeris[:, 3] + sigma_bounds[:, 3], alpha=0.3, color='blue', label='2σ bounds')
#     axes[3].plot(times, truth_ephemeris[:, 3], color='r', linestyle='--', label='True')
#     axes[3].set_ylabel('ζ_a (m)')
#     axes[3].set_ylim([-300, 100])
#     axes[3].legend()
#     axes[3].grid(True)
    
#     # Plot eta_a
#     axes[4].plot(t_vec, ephemeris[:, 4], label='Estimated', color='blue')
#     axes[4].fill_between(t_vec, ephemeris[:, 4] - sigma_bounds[:, 4], 
#                          ephemeris[:, 4] + sigma_bounds[:, 4], alpha=0.3, color='blue', label='2σ bounds')
#     axes[4].plot(times, truth_ephemeris[:, 4], color='r', linestyle='--', label='True')
#     axes[4].set_ylabel('η_a (m)')
#     axes[4].set_ylim([-100, 500])
#     axes[4].legend()
#     axes[4].grid(True)
    
#     # Plot theta_a
#     axes[5].plot(t_vec, ephemeris[:, 5], label='Estimated', color='blue')
#     axes[5].fill_between(t_vec, ephemeris[:, 5] - sigma_bounds[:, 5], 
#                          ephemeris[:, 5] + sigma_bounds[:, 5], alpha=0.3, color='blue', label='2σ bounds')
#     axes[5].plot(times, truth_ephemeris[:, 5], color='r', linestyle='--', label='True')
#     axes[5].set_xlabel('Time (s)')
#     axes[5].set_ylabel('θ_a (rad)')
#     axes[5].set_ylim([-4, 4])
#     axes[5].legend()
#     axes[5].grid(True)
    
#     plt.tight_layout()
#     plt.show()

# def test_ekf_nees():

#     """gen the plots as shown in the pdf"""
#     # assumed nominal trajectory
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
#     x_truth = x_0 + np.array([0, 1, 0, 0, 0, 0.1])

#     # constant control
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])

#     dt = 0.1
#     t = 0

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
#                         [0,0,0,0,0,0.01]]) / 20


#     # P_0 = np.eye(6) * 10
#     P_0 = np.diag([1.0, 1.0, 0.1, 1.0, 1.0, 0.1])

#     nees_sum = np.zeros([NUM_TESTING_STEPS, 1])
#     error_sum = np.zeros([NUM_TESTING_STEPS, 6])

#     for _ in range(0, NUM_TESTS):

#         ugv_truth = ugv_dynamics.Dynamical_UGV(x_truth[0:3])
#         uav_truth = uav_dynamics.Dynamical_UAV(x_truth[3:])
#         combo_truth = combined_system.CombinedSystem(ugv_truth, uav_truth)

#         #generate the truth model to run the nees testing on
#         tmt_times, tmt_states, tmt_measurement = combo_truth.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4])

#         # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
#         # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

#         ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#         uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#         combo = combined_system.CombinedSystem(ugv, uav)

#         ekf = filters.EKF(dt,
#                                 combo,
#                                 control,
#                                 Q_true,
#                                 R_true,
#                                 P_0,
#                                 x_0)  

#         tmt_measurement = np.insert(tmt_measurement, 0, np.zeros([5]), axis=1)
#         ekf.propagate(tmt_measurement)

#         x_ephem = np.array(ekf.x_ephem)

#         for step, cov in enumerate(ekf.P_ephem):

#             if(step == 0):
#                 continue

#             state_error = (x_ephem[step,:] - tmt_states[step - 1,:])

#             #normalize angles
#             if state_error[2] > math.pi:
#                 state_error[2] -= 2*math.pi
#             elif state_error[2] < -math.pi:
#                 state_error[2] += 2*math.pi
#             if state_error[5] > math.pi:
#                 state_error[5] -= 2*math.pi
#             elif state_error[5] < -math.pi:
#                 state_error[5] += 2*math.pi
            
#             error_sum[step-1, :] += (state_error)
#             nees_sum[step-1, :] += (state_error) @ np.linalg.inv(cov) @ np.transpose(state_error)

#     nees_sum = nees_sum / NUM_TESTS
#     error_sum = error_sum / 50

#     # #determine the chi2inv for upper and lower error bound
#     r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) / NUM_TESTS
#     r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) /  NUM_TESTS
   
#     fig, axes = plt.subplots(7, 1, figsize=(10, 12))

#     axes[0].plot(tmt_times, nees_sum[:, 0], marker='o', linestyle="none", color='blue')
#     axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
#     axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')

#     axes[1].plot(tmt_times, error_sum[:, 0], marker='o',  color='blue')
#     axes[2].plot(tmt_times, error_sum[:, 1], marker='o',  color='blue')
#     axes[3].plot(tmt_times, error_sum[:, 2], marker='o',  color='blue')
#     axes[4].plot(tmt_times, error_sum[:, 3], marker='o',  color='blue')
#     axes[5].plot(tmt_times, error_sum[:, 4], marker='o',  color='blue')
#     axes[6].plot(tmt_times, error_sum[:, 5], marker='o',  color='blue')
    
#     plt.tight_layout()
#     plt.show()


# def test_ekf_nis():
#     """gen the plots as shown in the pdf"""
#     # assumed nominal trajectory
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
#     x_truth = x_0 + np.array([0, 1, 0, 0, 0, 0.1])

#     # constant control
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])

#     dt = 0.1
#     t = 0

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
#                         [0,0,0,0,0,0.01]]) / 20


#     # P_0 = np.eye(6) * 10
#     P_0 = np.diag([1.0, 1.0, 0.1, 1.0, 1.0, 0.1])

#     nis_sum = np.zeros([NUM_TESTING_STEPS, 1])
#     error_sum = np.zeros([NUM_TESTING_STEPS, 5])

#     for _ in range(0, NUM_TESTS):

#         ugv_truth = ugv_dynamics.Dynamical_UGV(x_truth[0:3])
#         uav_truth = uav_dynamics.Dynamical_UAV(x_truth[3:])
#         combo_truth = combined_system.CombinedSystem(ugv_truth, uav_truth)

#         #generate the truth model to run the nees testing on
#         tmt_times, tmt_states, tmt_measurement = combo_truth.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4])

#         # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
#         # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

#         ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#         uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#         combo = combined_system.CombinedSystem(ugv, uav)

#         ekf = filters.EKF(dt,
#                                 combo,
#                                 control,
#                                 Q_true,
#                                 R_true,
#                                 P_0,
#                                 x_0)  

#         ekf.propagate(tmt_measurement)

#         tmt_measurement = tmt_measurement.transpose()

#         measure_ephem = np.array(ekf.measurement_ephem)

#         for step, cov in enumerate(ekf.P_pre_ephem):

#             if(step == 0):
#                 continue

#             measurement_error = (measure_ephem[step,:] - tmt_measurement[step,:])

#             #normalize angles
#             if measurement_error[0] > math.pi:
#                 measurement_error[0] -= 2*math.pi
#             elif measurement_error[0] < -math.pi:
#                 measurement_error[0] += 2*math.pi
#             if measurement_error[2] > math.pi:
#                 measurement_error[2] -= 2*math.pi
#             elif measurement_error[2] < -math.pi:
#                 measurement_error[2] += 2*math.pi

#             invoation_cov = ekf.H_ephem[step] @ cov @ np.transpose(ekf.H_ephem[step]) + ekf.R
            
#             error_sum[step, :] += (measurement_error)
#             nis_sum[step, :] += (measurement_error) @ np.linalg.inv(invoation_cov) @ np.transpose(measurement_error)

#     nis_sum = nis_sum / NUM_TESTS
#     error_sum = error_sum / 50

#     # #determine the chi2inv for upper and lower error bound
#     r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS
#     r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) /  NUM_TESTS
   
#     fig, axes = plt.subplots(6, 1, figsize=(10, 12))

#     axes[0].plot(tmt_times, nis_sum[:, 0], marker='o', linestyle="none", color='blue')
#     axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
#     axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')

#     axes[1].plot(tmt_times, error_sum[:, 0], marker='o',  color='blue')
#     axes[2].plot(tmt_times, error_sum[:, 1], marker='o',  color='blue')
#     axes[3].plot(tmt_times, error_sum[:, 2], marker='o',  color='blue')
#     axes[4].plot(tmt_times, error_sum[:, 3], marker='o',  color='blue')
#     axes[5].plot(tmt_times, error_sum[:, 4], marker='o',  color='blue')
    
#     plt.tight_layout()
#     plt.show()

# def test_q_continuous_vs_discrete():
#     """Compare measurement residuals for Q discrete vs continuous interpretation"""
    
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    
#     x_0 += np.array([0, 1, 0, 0, 0, 0.1])


    
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])
    
#     dt = 0.1
    
#     R_true = np.array([[0.0225,0,0,0,0],
#                         [0,64,0,0,0],
#                         [0,0,0.04,0,0],
#                         [0,0,0,36,0],
#                         [0,0,0,0,36]])
    
#     # The given Q from the problem
#     Q_given = np.array([[0.001,0,0,0,0,0],
#                         [0,0.001,0,0,0,0],
#                         [0,0,0.01,0,0,0],
#                         [0,0,0,0.001,0,0],
#                         [0,0,0,0,0.001,0],
#                         [0,0,0,0,0,0.01]])
    
#     NUM_STEPS = 100
    
#     # Generate ONE truth trajectory with process noise
#     print("Generating truth trajectory with process noise...")
#     ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#     combo_truth = combined_system.CombinedSystem(ugv, uav)
    
#     times, truth_states, truth_measurements = combo_truth.generate_truth_set(
#         dt, NUM_STEPS, R_true, control[0:2], control[2:4], process_noise=False
#     )
    
#     # Propagate states WITHOUT noise starting from x_0
#     print("Propagating nominal trajectory (no process noise)...")
#     ugv_nominal = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#     uav_nominal = uav_dynamics.Dynamical_UAV(x_0[3:])
#     combo_nominal = combined_system.CombinedSystem(ugv_nominal, uav_nominal)
    
#     nominal_states = [x_0]
#     for step in range(NUM_STEPS):
#         state = combo_nominal.step_nl_propagation(control, dt, state=nominal_states[-1])
#         nominal_states.append(state)
    
#     nominal_states = np.array(nominal_states[1:])  # Remove initial state, match truth_states length
    
#     # Convert nominal states to measurements
#     nominal_measurements = []
#     for state in nominal_states:
#         meas = combo_nominal.create_measurements_from_states(state=state, measurement_noise_cov=None)
#         nominal_measurements.append(meas)
    
#     nominal_measurements = np.array(nominal_measurements)
#     truth_measurements = truth_measurements.T  # Transpose to match shape
    
#     # Compute residuals: actual - predicted
#     residuals = truth_measurements - nominal_measurements
    
#     # Wrap angle residuals
#     for i in range(len(residuals)):
#         residuals[i, 0] = filters.angle_difference(truth_measurements[i, 0], nominal_measurements[i, 0])
#         residuals[i, 2] = filters.angle_difference(truth_measurements[i, 2], nominal_measurements[i, 2])
    
#     # Compute statistics
#     residual_means = np.mean(residuals, axis=0)
#     residual_stds = np.std(residuals, axis=0)
    
#     # Expected standard deviations from R (measurement noise only if Q is way off)
#     expected_stds_R = np.sqrt(np.diag(R_true))
    
#     print("\n" + "=" * 60)
#     print("RESIDUAL STATISTICS (Actual Measurements - Nominal Predictions)")
#     print("=" * 60)
#     print("Measurement | Mean Residual | Std Dev | Expected Std (from R)")
#     print("-" * 60)
#     meas_names = ['Bearing UGV→UAV', 'Range', 'Bearing UAV→UGV', 'UAV ζ', 'UAV η']
#     for i in range(5):
#         print(f"{meas_names[i]:16s} | {residual_means[i]:13.4f} | {residual_stds[i]:7.4f} | {expected_stds_R[i]:7.4f}")
    
#     # Create residual plots
#     fig, axes = plt.subplots(5, 1, figsize=(12, 14))
    
#     meas_labels = [
#         'Bearing from UGV to UAV (rad)',
#         'Range (m)',
#         'Bearing from UAV to UGV (rad)',
#         'UAV ζ position (m)',
#         'UAV η position (m)'
#     ]
    
#     for i in range(5):
#         axes[i].plot(times, residuals[:, i], 'b-', alpha=0.6, linewidth=1)
#         axes[i].axhline(0, color='k', linestyle='--', linewidth=1)
#         axes[i].axhline(residual_means[i], color='r', linestyle='--', linewidth=2, 
#                        label=f'Mean = {residual_means[i]:.4f}')
#         axes[i].fill_between(times, -residual_stds[i], residual_stds[i], 
#                             alpha=0.2, color='blue', label=f'±1σ = {residual_stds[i]:.4f}')
#         axes[i].set_ylabel(meas_labels[i])
#         axes[i].set_xlabel('Time (s)')
#         axes[i].legend(loc='upper right')
#         axes[i].grid(True, alpha=0.3)
#         axes[i].set_title(f'{meas_labels[i]} - Residual (Actual - Predicted)')
    
#     plt.tight_layout()
#     plt.show()
    
#     print("\n" + "=" * 60)
#     print("INTERPRETATION")
#     print("=" * 60)
#     print("If residuals have:")
#     print("  - Small bias (mean ≈ 0): Process model is unbiased")
#     print("  - Std close to √R: Minimal process noise effect (or Q is too small)")
#     print("  - Std >> √R: Significant process noise accumulation")
#     print("\nThese residuals show the accumulated effect of process noise")
#     print("over the propagation WITHOUT correction from measurements.")

# def test_estimate_R_from_data():
#     """Estimate measurement noise covariance R from ydata.csv"""
    
#     x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
#     x_0 += np.array([0, 1, 0, 0, 0, 0.1])
    
#     control = np.array([2, -math.pi / 18, 12, math.pi/25])
    
#     dt = 0.1
    
#     # Load measurement data
#     y_data = np.loadtxt(f"{os.getcwd()}/src/data/ydata.csv", delimiter=",")
#     t_vec = np.loadtxt(f"{os.getcwd()}/src/data/tvec.csv", delimiter=",")

#     y_data = y_data[:, 1:]
#     t_vec = t_vec[1:]
    
#     print("Loaded measurement data:")
#     print(f"  Shape: {y_data.shape}")
#     print(f"  Time steps: {len(t_vec)}")
    
#     # Generate clean nominal trajectory (no process noise)
#     ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
#     uav = uav_dynamics.Dynamical_UAV(x_0[3:])
#     combo = combined_system.CombinedSystem(ugv, uav)
    
#     nominal_states = [x_0]
#     for step in range(y_data.shape[1] - 1):  # -1 because we start at x_0
#         state = combo.step_nl_propagation(control, dt, state=nominal_states[-1])
#         nominal_states.append(state)
    
#     nominal_states = np.array(nominal_states)
    
#     # Convert nominal states to predicted measurements
#     predicted_measurements = []
#     for state in nominal_states:
#         meas = combo.create_measurements_from_states(state=state, measurement_noise_cov=None)
#         predicted_measurements.append(meas)
    
#     predicted_measurements = np.array(predicted_measurements).T  # Shape: (5, N)
    
#     # Compute measurement residuals (innovations)
#     residuals = y_data - predicted_measurements
    
#     # Handle angle wrapping for bearing measurements
#     for i in range(residuals.shape[1]):
#         residuals[0, i] = filters.angle_difference(y_data[0, i], predicted_measurements[0, i])
#         residuals[2, i] = filters.angle_difference(y_data[2, i], predicted_measurements[2, i])
    
#     # Compute sample covariance matrix
#     R_estimated = np.cov(residuals)
    
#     # Also compute individual variances
#     variances = np.var(residuals, axis=1)
#     std_devs = np.sqrt(variances)
    
#     # The "true" R from the problem
#     R_given = np.array([[0.0225,0,0,0,0],
#                         [0,64,0,0,0],
#                         [0,0,0.04,0,0],
#                         [0,0,0,36,0],
#                         [0,0,0,0,36]])
    
#     print("\n" + "=" * 70)
#     print("MEASUREMENT NOISE ANALYSIS")
#     print("=" * 70)
    
#     print("\nMeasurement residual statistics:")
#     print("-" * 70)
#     meas_names = ['Bearing UGV→UAV (rad)', 'Range (m)', 'Bearing UAV→UGV (rad)', 'UAV ζ (m)', 'UAV η (m)']
#     print(f"{'Measurement':<25} | {'Mean':<10} | {'Std Dev':<10} | {'Variance':<10}")
#     print("-" * 70)
#     for i in range(5):
#         mean_res = np.mean(residuals[i, :])
#         print(f"{meas_names[i]:<25} | {mean_res:>10.6f} | {std_devs[i]:>10.4f} | {variances[i]:>10.4f}")
    
#     print("\n" + "=" * 70)
#     print("ESTIMATED R (diagonal elements):")
#     print("-" * 70)
#     print(f"{'Measurement':<25} | {'Estimated':<12} | {'Given':<12} | {'Ratio':<10}")
#     print("-" * 70)
#     for i in range(5):
#         given_var = R_given[i, i]
#         ratio = variances[i] / given_var if given_var > 0 else float('inf')
#         print(f"{meas_names[i]:<25} | {variances[i]:>12.6f} | {given_var:>12.6f} | {ratio:>10.4f}")
    
#     print("\n" + "=" * 70)
#     print("ESTIMATED R MATRIX (full covariance):")
#     print("=" * 70)
#     print(R_estimated)
    
#     print("\n" + "=" * 70)
#     print("GIVEN R MATRIX:")
#     print("=" * 70)
#     print(R_given)
    
#     print("\n" + "=" * 70)
#     print("OFF-DIAGONAL CORRELATIONS:")
#     print("=" * 70)
#     # Compute correlation matrix
#     correlation = np.corrcoef(residuals)
#     print("Correlation matrix:")
#     print(correlation)
#     print("\nNote: Values close to 0 indicate measurements are uncorrelated")
#     print("      (which justifies a diagonal R matrix)")
    
#     # Create plots
#     fig, axes = plt.subplots(3, 2, figsize=(14, 10))
#     axes = axes.flatten()
    
#     for i in range(5):
#         axes[i].hist(residuals[i, :], bins=30, alpha=0.7, edgecolor='black', density=True)
        
#         # Overlay theoretical normal distribution
#         mu = np.mean(residuals[i, :])
#         sigma = std_devs[i]
#         x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
#         theoretical = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - mu)/sigma)**2)
#         axes[i].plot(x, theoretical, 'r-', linewidth=2, label=f'N({mu:.4f}, {sigma:.4f}²)')
        
#         axes[i].set_xlabel('Residual')
#         axes[i].set_ylabel('Density')
#         axes[i].set_title(meas_names[i])
#         axes[i].legend()
#         axes[i].grid(True, alpha=0.3)
    
#     # Remove the 6th subplot
#     fig.delaxes(axes[5])
    
#     plt.tight_layout()
#     plt.show()
    
#     # Plot residuals over time
#     fig2, axes2 = plt.subplots(5, 1, figsize=(12, 14))
    
#     for i in range(5):
#         axes2[i].plot(t_vec, residuals[i, :], 'b-', alpha=0.6, linewidth=1)
#         axes2[i].axhline(0, color='k', linestyle='--', linewidth=1)
#         axes2[i].axhline(np.mean(residuals[i, :]), color='r', linestyle='--', linewidth=2, 
#                         label=f'Mean = {np.mean(residuals[i, :]):.4f}')
#         axes2[i].fill_between(t_vec, -std_devs[i], std_devs[i], 
#                              alpha=0.2, color='blue', label=f'±1σ = {std_devs[i]:.4f}')
#         axes2[i].set_ylabel(meas_names[i])
#         axes2[i].set_xlabel('Time (s)')
#         axes2[i].legend(loc='upper right')
#         axes2[i].grid(True, alpha=0.3)
#         axes2[i].set_title(f'{meas_names[i]} - Measurement Residuals')
    
#     plt.tight_layout()
#     plt.show()
    
#     print("\n" + "=" * 70)
#     print("INTERPRETATION:")
#     print("=" * 70)
#     print("If estimated variances are close to given R:")
#     print("  → The given R is appropriate for this data")
#     print("If estimated >> given:")
#     print("  → Either measurement noise is larger than expected,")
#     print("     OR process noise is accumulating (Q effect)")
#     print("If estimated << given:")
#     print("  → Given R may be too conservative")
    
#     return R_estimated, R_given


def test_ekf_comprehensive():
    """Comprehensive EKF test: NEES, NIS, state residuals, and measurement residuals"""
    
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
                        [0,0,0,0,0,0.01]]) / 9
    
    P_0 = np.diag([1.0, 1.0, 0.01, 1.0, 1.0, 0.01])
    
    # NEES and NIS Testing
    nees_sum = np.zeros([NUM_TESTING_STEPS, 1])
    nis_sum = np.zeros([NUM_TESTING_STEPS -1, 1]) # missing one measurement
    
    #record single test for report
    single_measurements = None
    single_times = None
    single_states= None
    single_error = []
    single_two_sigma = []

    for test_num in range(NUM_TESTS):
        # ample initial state with uncertainty
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

        # Run filter from nominal x_0
        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav)
        
        ekf = filters.EKF(dt, combo, control, Q_true, R_true, P_0, x_0)
        
        tmt_measurement = np.insert(tmt_measurement, 0, np.zeros([5]), axis=1)
        ekf.propagate(tmt_measurement)
        
        x_ephem = np.array(ekf.x_ephem)
        measure_ephem = np.array(ekf.measurement_ephem)
        
        # Compute NEES
        for step, cov in enumerate(ekf.P_ephem):
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
            
            state_error = x_ephem[step,:] - tmt_states[step - 1,:]
            state_error[2] = filters.angle_difference(x_ephem[step,2], tmt_states[step-1,2])
            state_error[5] = filters.angle_difference(x_ephem[step,5], tmt_states[step-1,5])
            
            nees_sum[step-1, :] += state_error @ np.linalg.inv(cov) @ state_error
        
        # Compute NIS
        tmt_measurement_T = tmt_measurement.T
        for step in range(1, len(measure_ephem)):
            measurement_error = measure_ephem[step,:] - tmt_measurement_T[step+1,:]
            measurement_error[0] = filters.angle_difference(measure_ephem[step,0], tmt_measurement_T[step+1,0])
            measurement_error[2] = filters.angle_difference(measure_ephem[step,2], tmt_measurement_T[step+1,2])
            
            innovation_cov = ekf.H_ephem[step] @ ekf.P_pre_ephem[step] @ ekf.H_ephem[step].T + R_true
            nis_sum[step-1, :] += measurement_error @ np.linalg.inv(innovation_cov) @ measurement_error
    
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
    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)
    
    ekf_state = filters.EKF(dt, combo, control, Q_true, R_true, P_0, x_0)
    
    truth_measurements = np.insert(truth_measurements, 0, np.zeros([5]), axis=1)
    ekf_state.propagate(truth_measurements)
    
    x_ephem_state = np.array(ekf_state.x_ephem)
    
    # Compute state residuals
    state_residuals = np.zeros((NUM_TESTING_STEPS, 6))
    for i in range(NUM_TESTING_STEPS):
        state_residuals[i, :] = x_ephem_state[i+1, :] - truth_states[i, :]
        state_residuals[i, 2] = filters.angle_difference(x_ephem_state[i+1, 2], truth_states[i, 2])
        state_residuals[i, 5] = filters.angle_difference(x_ephem_state[i+1, 5], truth_states[i, 5])
    
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
    
    # Run filter on ydata
    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)
    
    x_0_data = x_0 + np.array([0, 1, 0, 0, 0, 0.1])
    ekf_meas = filters.EKF(dt, combo, control, Q_true, R_true, P_0, x_0_data)
    
    # Add NaN column at beginning
    y_data_with_nan = np.column_stack([np.zeros(5), y_data])
    ekf_meas.propagate(y_data_with_nan)
    
    measure_ephem_data = np.array(ekf_meas.measurement_ephem)
    
    # Compute measurement residuals
    meas_residuals = np.zeros((len(t_vec_data), 5))
    for i in range(len(t_vec_data)):
        meas_residuals[i, :] = y_data[:, i] - measure_ephem_data[i, :]
        meas_residuals[i, 0] = filters.angle_difference(y_data[0, i], measure_ephem_data[i, 0])
        meas_residuals[i, 2] = filters.angle_difference(y_data[2, i], measure_ephem_data[i, 2])
    
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
    ax1.set_title('Normalized Estimation Error Squared (NEES)')
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
    ax2.set_title('Normalized Innovation Squared (NIS)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    
    # Plot 3: State Residuals
    fig3, axes3 = plt.subplots(6, 1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    
    for i in range(6):
        axes3[i].plot(truth_times, state_residuals[:, i], 'b-', alpha=0.6, linewidth=1)
        axes3[i].axhline(0, color='k', linestyle='--', linewidth=1)
        mean_res = np.mean(state_residuals[:, i])
        std_res = np.std(state_residuals[:, i])
        axes3[i].axhline(mean_res, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_res:.4f}')
        axes3[i].fill_between(truth_times, -std_res, std_res, alpha=0.2, color='blue', label=f'±1σ = {std_res:.4f}')
        axes3[i].set_ylabel(state_labels[i])
        axes3[i].set_xlabel('Time (s)')
        axes3[i].legend(loc='upper right')
        axes3[i].grid(True, alpha=0.3)
        axes3[i].set_title(f'State Residual: {state_labels[i]} (Estimated - Truth)')
    
    plt.tight_layout()
    
    # Plot 4: Measurement Residuals
    fig4, axes4 = plt.subplots(5, 1, figsize=(12, 14))
    meas_labels = ['Bearing UGV→UAV (rad)', 'Range (m)', 'Bearing UAV→UGV (rad)', 'UAV ζ (m)', 'UAV η (m)']
    
    for i in range(5):
        axes4[i].plot(t_vec_data, meas_residuals[:, i], 'b-', alpha=0.6, linewidth=1)
        axes4[i].axhline(0, color='k', linestyle='--', linewidth=1)
        mean_res = np.mean(meas_residuals[:, i])
        std_res = np.std(meas_residuals[:, i])
        axes4[i].axhline(mean_res, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_res:.4f}')
        axes4[i].fill_between(t_vec_data, -std_res, std_res, alpha=0.2, color='blue', label=f'±1σ = {std_res:.4f}')
        axes4[i].set_ylabel(meas_labels[i])
        axes4[i].set_xlabel('Time (s)')
        axes4[i].legend(loc='upper right')
        axes4[i].grid(True, alpha=0.3)
        axes4[i].set_title(f'Measurement Residual: {meas_labels[i]} (Actual - Predicted)')
    
    fig5, axes5 = plt.subplots(5,1, figsize=(12, 14))
    meas_labels = ['Bearing UGV→UAV (rad)', 'Range (m)', 'Bearing UAV→UGV (rad)', 'UAV ζ (m)', 'UAV η (m)']
    for i in range(5):
        axes5[i].plot(single_times, single_measurements[i,:], "b-")
        axes5[i].set_ylabel(meas_labels[i])

    axes5[4].set_xlabel('Time (s)')
    axes5[0].set_title(f'Truth Model Noisy Measurements for EKF NEES / NIST Testing')

    fig6, axes6 = plt.subplots(6,1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    for i in range(6):
        axes6[i].plot(single_times, single_states[:,i], "b-")
        axes6[i].set_ylabel(state_labels[i])

    axes6[4].set_xlabel('Time (s)')
    axes6[0].set_title(f'Truth Model States for EKF NEES / NIS Testing ')

    fig7, axes7 = plt.subplots(6,1, figsize=(12, 14))
    state_labels = ['ζ_g (m)', 'η_g (m)', 'θ_g (rad)', 'ζ_a (m)', 'η_a (m)', 'θ_a (rad)']
    for i in range(6):
        axes7[i].plot(single_times, single_error[:,i], "r-", label="State Error")
        axes7[i].fill_between(single_times, single_error[:,i] - single_two_sigma[:,i], single_error[:,i] + single_two_sigma[:,i],  alpha=0.2, color='blue', label=f'±2σ')
        axes7[i].set_ylabel(state_labels[i])
        axes7[i].legend(loc="upper right")

    axes7[4].set_xlabel('Time (s)')
    axes7[0].set_title(f'Truth Model States for EKF NEES / NIS Testing ')


    plt.tight_layout()
    plt.show()


def cost_of_Q(Q_chol):
    '''finds the optimal Q to pass nees tests'''
    
    # Ensure Q is positive definite: Q = L·L^T
    Q = Q_chol @ Q_chol.T
    Q_np = np.array(Q.detach().tolist())
    
    # Same setup as test_ekf_comprehensive
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    control = np.array([2, -math.pi / 18, 12, math.pi/25])
    dt = 0.1
    
    R_true = np.array([[0.0225,0,0,0,0],
                        [0,64,0,0,0],
                        [0,0,0.04,0,0],
                        [0,0,0,36,0],
                        [0,0,0,0,36]])
    
    P_0 = np.diag([1.0, 1.0, 0.01, 1.0, 1.0, 0.01])
    
    nees_list = []

    for test_num in range(5):  # reduced a lot for speed
        
        x_0_truth = x_0 + np.random.multivariate_normal(np.zeros(6), P_0)
        
        ugv = ugv_dynamics.Dynamical_UGV(x_0_truth[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0_truth[3:])
        combo = combined_system.CombinedSystem(ugv, uav) 

        tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(
            dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4], process_noise=True
        )

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav)

        ekf = filters.EKF(dt, combo, control, Q_np, R_true, P_0, x_0)

        tmt_measurement = np.insert(tmt_measurement, 0, np.zeros([5]), axis=1)
        ekf.propagate(tmt_measurement)

        x_ephem = np.array(ekf.x_ephem)

        for step, cov in enumerate(ekf.P_ephem):
            if step == 0:
                continue

            state_error = x_ephem[step,:] - tmt_states[step - 1,:]
            
            state_error[2] = filters.angle_difference(x_ephem[step,2], tmt_states[step-1,2])
            state_error[5] = filters.angle_difference(x_ephem[step,5], tmt_states[step-1,5])
            
            nees = state_error @ np.linalg.inv(cov) @ state_error
            nees_list.append(nees)

    nees_tensor = torch.tensor(nees_list, dtype=torch.float32)
    
    expected_nees = 6.0
    
    r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 6)
    r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 6)
    
    loss_mean = (torch.mean(nees_tensor) - expected_nees)**2
    
    upper_violations = torch.relu(nees_tensor - r2_chi2)
    lower_violations = torch.relu(r1_chi2 - nees_tensor)
    loss_violations = torch.mean(upper_violations**2 + lower_violations**2)
    
    loss = loss_mean + 10.0 * loss_violations
    
    return loss


def test_optimize_Q_via_nees():
    from scipy.optimize import minimize
    
    # start far off so it has something to optimize
    Q_diag = np.array([0.1, 0.1, 0.01, 0.1, 0.1, 0.01])
    
    def cost_wrapper(q_diag_log):
        q_diag = np.exp(q_diag_log)
        Q_chol = np.diag(np.sqrt(q_diag))
        Q_chol_torch = torch.tensor(Q_chol, dtype=torch.float32)
        
        loss = cost_of_Q(Q_chol_torch)
        print(f"  Q_diag: {q_diag}, Loss: {loss.item():.4f}")
        return loss.item()
    
    result = minimize(
        cost_wrapper, 
        np.log(Q_diag),
        method='Powell',
        options={'maxiter': 10, 'disp': True}
    )
    
    Q_diag_optimized = np.exp(result.x)
    Q_optimized = np.diag(Q_diag_optimized)
    
    print("\nOptimized Q diagonal:", Q_diag_optimized)
    print("Optimized Q:\n", Q_optimized)
    
    return Q_optimized

if __name__ == "__main__":
    test_optimize_Q_via_nees()
    #test_ekf_comprehensive()
    # R_estimated, R_given = test_estimate_R_from_data()
    # test_q_continuous_vs_discrete()