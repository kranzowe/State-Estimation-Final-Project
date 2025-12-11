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

NUM_TESTS = 1
NUM_TESTING_STEPS = 1000
SIGNFICANCE_LEVEL = 0.01

def test_lkf():
    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    # x0 += np.array([0, 1, 0, 0, 0, 0.1])

    # constant control
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0


    # need to generate a nominal ephem and control
    nominal_ephemeris = [x0]
    nominal_controls = [control]
    times = [t]
    nominal_measurements = [combo.create_measurements_from_states()]
    while t <= 100-dt:
        combo.step_nl_propagation(control, dt)
        nominal_ephemeris.append(combo.current_state)
        nominal_controls.append(control)
        nominal_measurements.append(combo.create_measurements_from_states())
        t += dt
        times.append(t)

    #
    x0 += np.array([0, 1, 0, 0, 0, 0.1])
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
    dx_0 = np.zeros((6,))

    # now run the filter
    # reset class
    ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)
    lkf = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
                      combo,
                      Q_true,
                      R_true,
                      P_0,
                      dx_0)


    # y_data = np.loadtxt(f"../src/data/ydata.csv", delimiter=",")
    # t_vec = np.loadtxt(f"../src/data/tvec.csv", delimiter=",")

    # todo: temp usage of truth model
    ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)
    tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2],
                                                                      control[2:4], True)
    y_data = tmt_measurement



    lkf.propagate(y_data)

    # plot ephem  

    # ai helped me plot cuz eww
    nominal_ephemeris = np.array(nominal_ephemeris)
    tmt_states = np.concatenate((x0.reshape(1, 6), tmt_states), axis=0)
    true_dx = tmt_states - nominal_ephemeris
    # true_dx = np.array([0, 1, 0, 0, 0, 0.1])
    ephemeris = np.array(lkf.dx_ephem)
    P_history = np.array(lkf.P_ephem)
    
    # Extract 2-sigma bounds for each state (skip first timestep)
    sigma_bounds = np.zeros((len(P_history), 6))
    for i in range(len(P_history)):
        sigma_bounds[i, :] = 2 * np.sqrt(np.diag(P_history[i]))
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(6, 1, figsize=(10, 12))
    
    # Plot zeta_g
    axes[0].plot(times[:-1], ephemeris[:, 0], label='Estimated')
    axes[0].fill_between(times[:-1], ephemeris[:, 0] - sigma_bounds[:, 0], 
                         ephemeris[:, 0] + sigma_bounds[:, 0], alpha=0.3, label='2σ bounds')
    axes[0].plot(times, true_dx[:, 0], color='r', linestyle='--', label='True')
    axes[0].set_ylabel('dζ_g (m)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot eta_g
    axes[1].plot(times[:-1], ephemeris[:, 1], label='Estimated')
    axes[1].fill_between(times[:-1], ephemeris[:, 1] - sigma_bounds[:, 1], 
                         ephemeris[:, 1] + sigma_bounds[:, 1], alpha=0.3, label='2σ bounds')
    axes[1].plot(times, true_dx[:, 1], color='r', linestyle='--', label='True')
    axes[1].set_ylabel('dη_g (m)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot theta_g
    axes[2].plot(times[:-1], ephemeris[:, 2], label='Estimated')
    axes[2].fill_between(times[:-1], ephemeris[:, 2] - sigma_bounds[:, 2], 
                         ephemeris[:, 2] + sigma_bounds[:, 2], alpha=0.3, label='2σ bounds')
    axes[2].plot(times, true_dx[:, 2], color='r', linestyle='--', label='True')
    axes[2].set_ylabel('dθ_g (rad)')
    axes[2].legend()
    axes[2].grid(True)

    # Plot zeta_a
    axes[3].plot(times[:-1], ephemeris[:, 3], label='Estimated')
    axes[3].fill_between(times[:-1], ephemeris[:, 3] - sigma_bounds[:, 3], 
                         ephemeris[:, 3] + sigma_bounds[:, 3], alpha=0.3, label='2σ bounds')
    axes[3].plot(times, true_dx[:, 3], color='r', linestyle='--', label='True')
    axes[3].set_ylabel('dζ_a (m)')
    axes[3].legend()
    axes[3].grid(True)
    
    # Plot eta_a
    axes[4].plot(times[:-1], ephemeris[:, 4], label='Estimated')
    axes[4].fill_between(times[:-1], ephemeris[:, 4] - sigma_bounds[:, 4], 
                         ephemeris[:, 4] + sigma_bounds[:, 4], alpha=0.3, label='2σ bounds')
    axes[4].plot(times, true_dx[:, 4], color='r', linestyle='--', label='True')
    axes[4].set_ylabel('dη_a (m)')
    axes[4].legend()
    axes[4].grid(True)
    
    # Plot theta_a
    axes[5].plot(times[:-1], ephemeris[:, 5], label='Estimated')
    axes[5].fill_between(times[:-1], ephemeris[:, 5] - sigma_bounds[:, 5], 
                         ephemeris[:, 5] + sigma_bounds[:, 5], alpha=0.3, label='2σ bounds')
    axes[5].plot(times, true_dx[:, 5], color='r', linestyle='--', label='True')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('dθ_a (rad)')
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()

def test_lkf_nees():
    np.random.seed(0)
    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])

    # constant control
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    dx_0 = np.zeros((6,))


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
    nominal_ephemeris = [x_0]
    nominal_controls = [control]
    times = [t]
    nominal_measurements = [combo.create_measurements_from_states()]
    while t < NUM_TESTING_STEPS * dt:
        combo.step_nl_propagation(control, dt)
        nominal_ephemeris.append(combo.current_state)
        nominal_controls.append(control)
        nominal_measurements.append(combo.create_measurements_from_states())
        t += dt
        times.append(t)

    # now apply perturbation
    x_0 += np.array([0, 1, 0, 0, 0, 0.1])

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

    nees_sum = np.zeros([NUM_TESTING_STEPS, 6])

    for _ in range(0, NUM_TESTS):

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav)

        lkf: filters.LKF = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
                      combo,
                      Q_true,
                      R_true,
                      P_0,
                      dx_0)  

        #generate the truth model to run the nees testing on
        tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2], control[2:4])

        # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
        # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

        lkf.propagate(tmt_measurement)

        x_ephem = np.array(lkf.dx_ephem) + np.array(nominal_ephemeris)[1:NUM_TESTING_STEPS+1]

        for step, cov in enumerate(lkf.P_ephem):
            diff = x_ephem[step,:] - tmt_states[step,:]
            diff[2] = filters.wrap_angle(diff[2])
            diff[5] = filters.wrap_angle(diff[5])
            nees_sum[step,:] += diff @ np.linalg.inv(cov) @ np.transpose(diff)
        # todo: temp debugging plot
        # Create figure with 6 subplots
        fig, axes = plt.subplots(6, 1, figsize=(10, 12))

        # Plot zeta_g
        axes[0].plot(tmt_times, x_ephem[:, 0], label='Estimated')
        axes[0].plot(tmt_times, tmt_states[:, 0], linestyle='--', label='true')
        axes[0].set_ylabel('dζ_g (m)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot eta_g
        axes[1].plot(tmt_times, x_ephem[:, 1], label='Estimated')
        axes[1].plot(tmt_times, tmt_states[:, 1], linestyle='--', label='true')
        axes[1].set_ylabel('dη_g (m)')
        axes[1].legend()
        axes[1].grid(True)

        # Plot theta_g
        axes[2].plot(tmt_times, x_ephem[:, 2], label='Estimated')
        axes[2].plot(tmt_times, tmt_states[:, 2], linestyle='--', label='true')
        axes[2].set_ylabel('dθ_g (rad)')
        axes[2].legend()
        axes[2].grid(True)

        # Plot zeta_a
        axes[3].plot(tmt_times, x_ephem[:, 3], label='Estimated')
        axes[3].plot(tmt_times, tmt_states[:, 3], linestyle='--', label='true')
        axes[3].set_ylabel('dζ_a (m)')
        axes[3].legend()
        axes[3].grid(True)

        # Plot eta_a
        axes[4].plot(tmt_times, x_ephem[:, 4], label='Estimated')
        axes[4].plot(tmt_times, tmt_states[:, 4], linestyle='--', label='true')
        axes[4].set_ylabel('dη_a (m)')
        axes[4].legend()
        axes[4].grid(True)

        # Plot theta_a
        axes[5].plot(tmt_times, x_ephem[:, 5], label='Estimated')
        axes[5].plot(tmt_times, tmt_states[:, 5], linestyle='--', label='true')
        axes[5].set_xlabel('Time (s)')
        axes[5].set_ylabel('dθ_a (rad)')
        axes[5].legend()
        axes[5].grid(True)


    nees_sum = nees_sum / NUM_TESTS

    # #determine the chi2inv for upper and lower error bound
    r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) / NUM_TESTS
    r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 6 * NUM_TESTS) /  NUM_TESTS
   
    fig, axes = plt.subplots(1, 1, figsize=(10, 12))

    axes.plot(tmt_times, nees_sum[:, 0], marker='o', linestyle="none", color='blue')
    axes.plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
    axes.plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')
    axes.grid(True)


    plt.tight_layout()
    plt.show()



def test_lfk_nis():
    np.random.seed(0)
    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x_0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])

    # constant control
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x_0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    dx_0 = np.zeros((6,))

    # get nominal trajectory
    nominal_ephemeris = [x_0]
    nominal_controls = [control]
    times = [t]
    nominal_measurements = [combo.create_measurements_from_states()]
    while t < NUM_TESTING_STEPS * dt:
        combo.step_nl_propagation(control, dt)
        nominal_ephemeris.append(combo.current_state)
        nominal_controls.append(control)
        nominal_measurements.append(combo.create_measurements_from_states())
        t += dt
        times.append(t)

    # now apply perturbation
    x_0 += np.array([0, 1, 0, 0, 0, 0.1])

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

    nis_sum = np.zeros([NUM_TESTING_STEPS, 6])
    # todo temp
    tmt_times = []
    y_ephem = []
    tmt_measurement = []
    for _ in range(0, NUM_TESTS):

        ugv = ugv_dynamics.Dynamical_UGV(x_0[0:3])
        uav = uav_dynamics.Dynamical_UAV(x_0[3:])
        combo = combined_system.CombinedSystem(ugv, uav)

        lkf: filters.LKF = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
                                       combo,
                                       Q_true,
                                       R_true,
                                       P_0,
                                       dx_0)

        # generate the truth model to run the nees testing on
        tmt_times, tmt_states, tmt_measurement = combo.generate_truth_set(dt, NUM_TESTING_STEPS, R_true, control[0:2],
                                                                          control[2:4])

        # y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
        # t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

        lkf.propagate(tmt_measurement)

        y_ephem = np.array(lkf.dy_ephem) + np.array(nominal_measurements)[1:NUM_TESTING_STEPS]
        # x_ephem = np.array(lkf.dx_ephem) + np.array(nominal_ephemeris)[1:NUM_TESTING_STEPS + 1]
        H_ephem = np.array(lkf.H_ephem)
        P_pre_ephem = np.array(lkf.P_pre_ephem)

        for step, (P_pre, H) in enumerate(zip(P_pre_ephem, H_ephem)):
            diff = y_ephem[step,:] - np.transpose(tmt_measurement[:, step+1])
            diff[0] = filters.wrap_angle(diff[0])
            diff[2] = filters.wrap_angle(diff[2])

            S = H @ P_pre @ np.transpose(H) + lkf.R
            nis_sum[step, :] += (diff) @ np.linalg.inv(S) @ np.transpose(diff)

    nis_sum = nis_sum / NUM_TESTS

    # #determine the chi2inv for upper and lower error bound
    r1_chi2 = chi2.ppf(SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS
    r2_chi2 = chi2.ppf(1 - SIGNFICANCE_LEVEL / 2, 5 * NUM_TESTS) / NUM_TESTS

    # todo: temp debugging plot
    # Create figure with 6 subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))

    # Plot zeta_g
    axes[0].plot(tmt_times[1:], y_ephem[:, 0], label='Estimated')
    axes[0].plot(tmt_times, tmt_measurement[0, :], linestyle='--', label='true')
    axes[0].set_ylabel('dζ_g (m)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot eta_g
    axes[1].plot(tmt_times[1:], y_ephem[:, 1], label='Estimated')
    axes[1].plot(tmt_times, tmt_measurement[1, :], linestyle='--', label='true')
    axes[1].set_ylabel('dη_g (m)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot theta_g
    axes[2].plot(tmt_times[1:], y_ephem[:, 2], label='Estimated')
    axes[2].plot(tmt_times, tmt_measurement[2, :], linestyle='--', label='true')
    axes[2].set_ylabel('dθ_g (rad)')
    axes[2].legend()
    axes[2].grid(True)

    # Plot zeta_a
    axes[3].plot(tmt_times[1:], y_ephem[:, 3], label='Estimated')
    axes[3].plot(tmt_times, tmt_measurement[3, :], linestyle='--', label='true')
    axes[3].set_ylabel('dζ_a (m)')
    axes[3].legend()
    axes[3].grid(True)

    # Plot eta_a
    axes[4].plot(tmt_times[1:], y_ephem[:, 4], label='Estimated')
    axes[4].plot(tmt_times, tmt_measurement[4, :], linestyle='--', label='true')
    axes[4].set_ylabel('dη_a (m)')
    axes[4].legend()
    axes[4].grid(True)

    fig, axes = plt.subplots(1, 1, figsize=(10, 12))

    axes.plot(tmt_times, nis_sum[:, 0], marker='o', linestyle="none", color='blue')
    axes.plot(tmt_times, np.ones(len(tmt_times)) * r1_chi2, linestyle='--', color='red')
    axes.plot(tmt_times, np.ones(len(tmt_times)) * r2_chi2, linestyle='--', color='red')
    axes.grid(True)

    plt.tight_layout()
    plt.show()