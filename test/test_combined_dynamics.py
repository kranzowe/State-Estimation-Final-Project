'''tests the dynamics'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import src.ugv_dynamics as ugv_dynamics
import src.uav_dynamics as uav_dynamics
import src.combined_system as combined_system
import math

import matplotlib.pyplot as plt


def test_nl_combined_prop_plots_progress_report_1():
    """gen the plots as shown in the pdf"""
    perturb_x0 = np.array([0, 1, 0, 0, 0, 0.1])
    x0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2]) + perturb_x0
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    ephemeris = [x0]
    times = [t]
    while t <=100:
        combo.step_nl_propagation(control, dt)
        ephemeris.append(combo.current_state)
        t += dt
        times.append(t)

    # ai helped me plot cuz eww
    ephemeris = np.array(ephemeris)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(6, 1, figsize=(10, 8))
    
    # Plot zeta_g
    axes[0].plot(times, ephemeris[:, 0])
    axes[0].set_ylabel('ζ_g (m)')
    axes[0].grid(True)
    
    # Plot eta_gugv = ugv_dynamics.Dynamical_UGV(x0)
    axes[1].plot(times, ephemeris[:, 1])
    axes[1].set_ylabel('η_g (m)')
    axes[1].grid(True)
    
    # Plot theta_g
    axes[2].plot(times, ephemeris[:, 2])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('θ_g (rad)')
    axes[2].grid(True)

    axes[3].plot(times, ephemeris[:, 3])
    axes[3].set_ylabel('ζ_a (m)')
    axes[3].grid(True)
    
    # Plot eta_gugv = ugv_dynamics.Dynamical_UGV(x0)
    axes[4].plot(times, ephemeris[:, 4])
    axes[4].set_ylabel('η_a (m)')
    axes[4].grid(True)
    
    # Plot theta_g
    axes[5].plot(times, ephemeris[:, 5])
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('θ_a (rad)')
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.show()

def test_nl_combined_prop_plots_progress_report_1():
    """gen the plots as shown in the pdf"""
    perturb_x0 = np.array([0, 1, 0, 0, 0, 0.1])
    x0 = np.array([10, 0, math.pi/2, -60, 0, -math.pi/2]) + perturb_x0
    control = np.array([2, -math.pi / 18, 12, math.pi/25])

    ugv = ugv_dynamics.Dynamical_UGV(x0[0:3])
    uav = uav_dynamics.Dynamical_UAV(x0[3:])
    combo = combined_system.CombinedSystem(ugv, uav)

    dt = 0.1
    t = 0

    ephemeris = [x0]
    times = [t]
    measurements = [combo.create_measurements_from_states()]
    while t <=100:
        combo.step_nl_propagation(control, dt)
        ephemeris.append(combo.current_state)
        measurements.append(combo.create_measurements_from_states())
        t += dt
        times.append(t)

    # ai helped me plot cuz eww
    ephemeris = np.array(ephemeris)
    measurements = np.array(measurements)
    
    # Create figure with 3 subplots
    fig, axes_1 = plt.subplots(6, 1, figsize=(10, 8))
    
    # Plot zeta_g
    axes_1[0].plot(times, ephemeris[:, 0])
    axes_1[0].set_ylabel('ζ_g (m)')
    axes_1[0].grid(True)
    
    # Plot eta_gugv = ugv_dynamics.Dynamical_UGV(x0)
    axes_1[1].plot(times, ephemeris[:, 1])
    axes_1[1].set_ylabel('η_g (m)')
    axes_1[1].grid(True)
    
    # Plot theta_g
    axes_1[2].plot(times, ephemeris[:, 2])
    axes_1[2].set_xlabel('Time (s)')
    axes_1[2].set_ylabel('θ_g (rad)')
    axes_1[2].grid(True)

    axes_1[3].plot(times, ephemeris[:, 3])
    axes_1[3].set_ylabel('ζ_a (m)')
    axes_1[3].grid(True)
    
    # Plot eta_gugv = ugv_dynamics.Dynamical_UGV(x0)
    axes_1[4].plot(times, ephemeris[:, 4])
    axes_1[4].set_ylabel('η_a (m)')
    axes_1[4].grid(True)
    
    # Plot theta_g
    axes_1[5].plot(times, ephemeris[:, 5])
    axes_1[5].set_xlabel('Time (s)')
    axes_1[5].set_ylabel('θ_a (rad)')
    axes_1[5].grid(True)
    
    fig, axes_2 = plt.subplots(5, 1, figsize=(10, 8))

    #plot
    axes_2[0].plot(times, measurements[:, 0])
    axes_2[0].set_ylabel('θ_1 (rad)')
    axes_2[0].grid(True)

    axes_2[1].plot(times, measurements[:, 1])
    axes_2[1].set_ylabel('D (m)')
    axes_2[1].grid(True)

    axes_2[2].plot(times, measurements[:, 2])
    axes_2[2].set_ylabel('θ_2 (rad)')
    axes_2[2].grid(True)

    axes_2[3].plot(times, measurements[:, 3])
    axes_2[3].set_ylabel('ζ_a (m)')
    axes_2[3].grid(True)

    axes_2[4].plot(times, measurements[:, 4])
    axes_2[4].set_ylabel('η_a (m)')
    axes_2[4].set_xlabel('Time (s)')
    axes_2[4].grid(True)

    plt.tight_layout()
    plt.show()


    




    
