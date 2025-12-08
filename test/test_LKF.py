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

import matplotlib.pyplot as plt


def test_lkf():
    """gen the plots as shown in the pdf"""
    # assumed nominal trajectory
    x0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2])
    #x0 += np.array([0, 1, 0, 0, 0, 0.1])

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
    while t < 100:
        combo.step_nl_propagation(control, dt)
        nominal_ephemeris.append(combo.current_state)
        nominal_controls.append(control)
        nominal_measurements.append(combo.create_measurements_from_states())
        t += dt
        times.append(t)

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

    P_0 = np.eye(6) * 1000000
    dx_0 = np.zeros((6,))

    # now run the filter
    lkf = filters.LKF(dt, nominal_ephemeris, nominal_controls, nominal_measurements,
                      combo,
                      Q_true,
                      R_true,
                      P_0,
                      dx_0)  

    

    y_data = np.loadtxt("src\data\ydata.csv", delimiter=",")
    t_vec = np.loadtxt(r"src\data\tvec.csv", delimiter=",")

    lkf.propagate(y_data)

    # plot ephem  

    # ai helped me plot cuz eww
    true_dx = np.array([0, 1, 0, 0, 0, 0.1])
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
    axes[0].axhline(y=true_dx[0], color='r', linestyle='--', label='True')
    axes[0].set_ylabel('dζ_g (m)')
    axes[0].set_ylim([-5, 5])
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot eta_g
    axes[1].plot(times[:-1], ephemeris[:, 1], label='Estimated')
    axes[1].fill_between(times[:-1], ephemeris[:, 1] - sigma_bounds[:, 1], 
                         ephemeris[:, 1] + sigma_bounds[:, 1], alpha=0.3, label='2σ bounds')
    axes[1].axhline(y=true_dx[1], color='r', linestyle='--', label='True')
    axes[1].set_ylabel('dη_g (m)')
    axes[1].set_ylim([-5, 5])
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot theta_g
    axes[2].plot(times[:-1], ephemeris[:, 2], label='Estimated')
    axes[2].fill_between(times[:-1], ephemeris[:, 2] - sigma_bounds[:, 2], 
                         ephemeris[:, 2] + sigma_bounds[:, 2], alpha=0.3, label='2σ bounds')
    axes[2].axhline(y=true_dx[2], color='r', linestyle='--', label='True')
    axes[2].set_ylabel('dθ_g (rad)')
    axes[2].set_ylim([-0.5, 0.5])
    axes[2].legend()
    axes[2].grid(True)

    # Plot zeta_a
    axes[3].plot(times[:-1], ephemeris[:, 3], label='Estimated')
    axes[3].fill_between(times[:-1], ephemeris[:, 3] - sigma_bounds[:, 3], 
                         ephemeris[:, 3] + sigma_bounds[:, 3], alpha=0.3, label='2σ bounds')
    axes[3].axhline(y=true_dx[3], color='r', linestyle='--', label='True')
    axes[3].set_ylabel('dζ_a (m)')
    axes[3].set_ylim([-5, 5])
    axes[3].legend()
    axes[3].grid(True)
    
    # Plot eta_a
    axes[4].plot(times[:-1], ephemeris[:, 4], label='Estimated')
    axes[4].fill_between(times[:-1], ephemeris[:, 4] - sigma_bounds[:, 4], 
                         ephemeris[:, 4] + sigma_bounds[:, 4], alpha=0.3, label='2σ bounds')
    axes[4].axhline(y=true_dx[4], color='r', linestyle='--', label='True')
    axes[4].set_ylabel('dη_a (m)')
    axes[4].set_ylim([-5, 5])
    axes[4].legend()
    axes[4].grid(True)
    
    # Plot theta_a
    axes[5].plot(times[:-1], ephemeris[:, 5], label='Estimated')
    axes[5].fill_between(times[:-1], ephemeris[:, 5] - sigma_bounds[:, 5], 
                         ephemeris[:, 5] + sigma_bounds[:, 5], alpha=0.3, label='2σ bounds')
    axes[5].axhline(y=true_dx[5], color='r', linestyle='--', label='True')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('dθ_a (rad)')
    axes[5].set_ylim([-0.5, 0.5])
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()