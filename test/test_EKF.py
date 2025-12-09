'''tests the linearized kalman filter'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import src.ugv_dynamics as ugv_dynamics
import src.uav_dynamics as uav_dynamics
import src.combined_system as combined_system
from src import filters
import math
from scipy.stats import chi2


import matplotlib.pyplot as plt

NUM_TESTS = 50
NUM_TESTING_STEPS = 100
SIGNFICANCE_LEVEL = 0.01


def test_ekf():
    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    x_0 += np.array([0, 1, 0, 0, 0, 0.1])

    # constant control
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    ## generate truth ephem
    truth_ephemeris = [x_0]
    times = [t]
    while t <=100:
        combo.step_nl_propagation(control, dt)
        truth_ephemeris.append(combo.current_state)
        t += dt
        times.append(t)

    # ai helped me plot cuz eww
    truth_ephemeris = np.array(truth_ephemeris)

    #
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
                        [0,0,0,0,0,0.01]])


    P_0 = np.eye(6) * 10

    # now run the filter
    ekf = filters.EKF(dt,
                      combo,
                      control,
                      Q_true,
                      R_true,
                      P_0,
                      x_0)  

    y_data = np.loadtxt(f"{os.getcwd()}/src/data/ydata.csv", delimiter=",")
    t_vec = np.loadtxt(f"{os.getcwd()}/src/data/tvec.csv", delimiter=",")

    ekf.propagate(y_data)

    # plot ephem  

    # ai helped me plot cuz eww
    ephemeris = np.array(ekf.x_ephem)
    P_history = np.array(ekf.P_ephem)
    
    # Extract 2-sigma bounds for each state
    sigma_bounds = np.zeros((len(P_history), 6))
    for i in range(len(P_history)):
        sigma_bounds[i, :] = 2 * np.sqrt(np.diag(P_history[i]))
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    
    # Plot zeta_g
    axes[0].plot(t_vec, ephemeris[:, 0], label='Estimated', color='blue')
    axes[0].fill_between(t_vec, ephemeris[:, 0] - sigma_bounds[:, 0], 
                         ephemeris[:, 0] + sigma_bounds[:, 0], alpha=0.3, color='blue', label='2σ bounds')
    axes[0].plot(times, truth_ephemeris[:, 0], color='r', linestyle='--', label='True')
    axes[0].set_ylabel('ζ_g (m)')
    axes[0].set_ylim([-100, 300])
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot eta_g
    axes[1].plot(t_vec, ephemeris[:, 1], label='Estimated', color='blue')
    axes[1].fill_between(t_vec, ephemeris[:, 1] - sigma_bounds[:, 1], 
                         ephemeris[:, 1] + sigma_bounds[:, 1], alpha=0.3, color='blue', label='2σ bounds')
    axes[1].plot(times, truth_ephemeris[:, 1], color='r', linestyle='--', label='True')
    axes[1].set_ylabel('η_g (m)')
    axes[1].set_ylim([-50, 50])
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot theta_g
    axes[2].plot(t_vec, ephemeris[:, 2], label='Estimated', color='blue')
    axes[2].fill_between(t_vec, ephemeris[:, 2] - sigma_bounds[:, 2], 
                         ephemeris[:, 2] + sigma_bounds[:, 2], alpha=0.3, color='blue', label='2σ bounds')
    axes[2].plot(times, truth_ephemeris[:, 2], color='r', linestyle='--', label='True')
    axes[2].set_ylabel('θ_g (rad)')
    axes[2].set_ylim([-4, 4])
    axes[2].legend()
    axes[2].grid(True)

    # Plot zeta_a
    axes[3].plot(t_vec, ephemeris[:, 3], label='Estimated', color='blue')
    axes[3].fill_between(t_vec, ephemeris[:, 3] - sigma_bounds[:, 3], 
                         ephemeris[:, 3] + sigma_bounds[:, 3], alpha=0.3, color='blue', label='2σ bounds')
    axes[3].plot(times, truth_ephemeris[:, 3], color='r', linestyle='--', label='True')
    axes[3].set_ylabel('ζ_a (m)')
    axes[3].set_ylim([-300, 100])
    axes[3].legend()
    axes[3].grid(True)
    
    # Plot eta_a
    axes[4].plot(t_vec, ephemeris[:, 4], label='Estimated', color='blue')
    axes[4].fill_between(t_vec, ephemeris[:, 4] - sigma_bounds[:, 4], 
                         ephemeris[:, 4] + sigma_bounds[:, 4], alpha=0.3, color='blue', label='2σ bounds')
    axes[4].plot(times, truth_ephemeris[:, 4], color='r', linestyle='--', label='True')
    axes[4].set_ylabel('η_a (m)')
    axes[4].set_ylim([-100, 500])
    axes[4].legend()
    axes[4].grid(True)
    
    # Plot theta_a
    axes[5].plot(t_vec, ephemeris[:, 5], label='Estimated', color='blue')
    axes[5].fill_between(t_vec, ephemeris[:, 5] - sigma_bounds[:, 5], 
                         ephemeris[:, 5] + sigma_bounds[:, 5], alpha=0.3, color='blue', label='2σ bounds')
    axes[5].plot(times, truth_ephemeris[:, 5], color='r', linestyle='--', label='True')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('θ_a (rad)')
    axes[5].set_ylim([-4, 4])
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()

def test_ekf_nees():

    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    x_0 += np.array([0, 1, 0, 0, 0, 0.1])

    # constant control
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    # ## generate truth ephem
    # truth_ephemeris = [x_0]
    # times = [t]
    # while t <=100:
    #     combo.step_nl_propagation(control, dt)
    #     truth_ephemeris.append(combo.current_state)
    #     t += dt
    #     times.append(t)

    # # ai helped me plot cuz eww
    # truth_ephemeris = np.array(truth_ephemeris)

    #
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
                        [0,0,0,0,0,0.01]])


    P_0 = np.eye(6) * 10

    nees_sum = np.zeros([NUM_TESTING_STEPS, 1])
    error_sum = np.zeros([NUM_TESTING_STEPS, 6])

    for _ in range(0, NUM_TESTS):

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav) 

        #generate the truth model to run the nees testing on
        tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4])

        # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
        # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav)

        ekf = filters.EKF(dt,
                                combo,
                                control,
                                Q_true,
                                R_true,
                                P_0,
                                x_0)  

        tmt_measurement = np.insert(tmt_measurement, 0, np.zeros([5]), axis=1)
        ekf.propagate(tmt_measurement)

        x_ephem = np.array(ekf.x_ephem)

        for step, cov in enumerate(ekf.P_ephem):

            if(step == 0):
                continue

            state_error = (x_ephem[step,:] - tmt_states[step - 1,:])

            #normalize angles
            if state_error[2] > math.pi:
                state_error[2] -= 2*math.pi
            elif state_error[2] < -math.pi:
                state_error[2] += 2*math.pi
            if state_error[5] > math.pi:
                state_error[5] -= 2*math.pi
            elif state_error[5] < -math.pi:
                state_error[5] += 2*math.pi
            
            error_sum[step-1, :] += (state_error)
            nees_sum[step-1, :] += (state_error) @ np.linalg.inv(cov) @ np.transpose(state_error)

    nees_sum = nees_sum / NUM_TESTS
    error_sum = error_sum / 50

    # #determine the chi2inv for upper and lower error bound
    r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) / NUM_TESTS
    r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) /  NUM_TESTS
   
    fig, axes = plt.subplots(7, 1, figsize=(10, 12))

    axes[0].plot(tmt_times, nees_sum[:, 0], marker='o', linestyle="none", color='blue')
    axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
    axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')

    axes[1].plot(tmt_times, error_sum[:, 0], marker='o',  color='blue')
    axes[2].plot(tmt_times, error_sum[:, 1], marker='o',  color='blue')
    axes[3].plot(tmt_times, error_sum[:, 2], marker='o',  color='blue')
    axes[4].plot(tmt_times, error_sum[:, 3], marker='o',  color='blue')
    axes[5].plot(tmt_times, error_sum[:, 4], marker='o',  color='blue')
    axes[6].plot(tmt_times, error_sum[:, 5], marker='o',  color='blue')
    
    plt.tight_layout()
    plt.show()


def test_ekf_nis():

    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    x_0 += np.array([0, 1, 0, 0, 0, 0.1])

    # constant control
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    # ## generate truth ephem
    # truth_ephemeris = [x_0]
    # times = [t]
    # while t <=100:
    #     combo.step_nl_propagation(control, dt)
    #     truth_ephemeris.append(combo.current_state)
    #     t += dt
    #     times.append(t)

    # # ai helped me plot cuz eww
    # truth_ephemeris = np.array(truth_ephemeris)

    #
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
                        [0,0,0,0,0,0.01]])


    P_0 = np.eye(6) * 10

    nis_sum = np.zeros([NUM_TESTING_STEPS, 1])
    error_sum = np.zeros([NUM_TESTING_STEPS, 5])

    for _ in range(0, NUM_TESTS):

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav) 

        #generate the truth model to run the nees testing on
        tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4])

        # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
        # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav)

        ekf = filters.EKF(dt,
                                combo,
                                control,
                                Q_true,
                                R_true,
                                P_0,
                                x_0)  

        ekf.propagate(tmt_measurement)

        tmt_measurement = tmt_measurement.transpose()

        measure_ephem = np.array(ekf.measurement_ephem)

        for step, cov in enumerate(ekf.P_pre_ephem):

            if(step == 0):
                continue

            measurement_error = (measure_ephem[step,:] - tmt_measurement[step,:])

            #normalize angles
            if measurement_error[0] > math.pi:
                measurement_error[0] -= 2*math.pi
            elif measurement_error[0] < -math.pi:
                measurement_error[0] += 2*math.pi
            if measurement_error[2] > math.pi:
                measurement_error[2] -= 2*math.pi
            elif measurement_error[2] < -math.pi:
                measurement_error[2] += 2*math.pi

            invoation_cov = ekf.H_ephem[step] @ cov @ np.transpose(ekf.H_ephem[step]) + ekf.R
            
            error_sum[step, :] += (measurement_error)
            nis_sum[step, :] += (measurement_error) @ np.linalg.inv(invoation_cov) @ np.transpose(measurement_error)

    nis_sum = nis_sum / NUM_TESTS
    error_sum = error_sum / 50

    # #determine the chi2inv for upper and lower error bound
    r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS
    r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) /  NUM_TESTS
   
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))

    axes[0].plot(tmt_times, nis_sum[:, 0], marker='o', linestyle="none", color='blue')
    axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
    axes[0].plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')

    axes[1].plot(tmt_times, error_sum[:, 0], marker='o',  color='blue')
    axes[2].plot(tmt_times, error_sum[:, 1], marker='o',  color='blue')
    axes[3].plot(tmt_times, error_sum[:, 2], marker='o',  color='blue')
    axes[4].plot(tmt_times, error_sum[:, 3], marker='o',  color='blue')
    axes[5].plot(tmt_times, error_sum[:, 4], marker='o',  color='blue')
    
    plt.tight_layout()
    plt.show()