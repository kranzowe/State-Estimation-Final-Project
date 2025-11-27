'''tests the dynamics'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import src.ugv_dynamics as ugv_dynamics
import math

import matplotlib.pyplot as plt


def test_nl_ugv_prop_plots_progress_report_1():
    """gen the plots as shown in the pdf"""
    perturb_x0 = np.array([0, 1, 0])
    x0 = np.array([10, 0, math.pi/2]) + perturb_x0
    control = np.array([2, -math.pi / 18])

    ugv = ugv_dynamics.Dynamical_UGV(x0)

    dt = 0.1
    t = 0

    ephemeris = [x0]
    times = [t]
    while t <=100:
        ugv.step_nl_propagation(control, dt)
        ephemeris.append(ugv.current_state)
        t += dt
        times.append(t)

    # ai helped me plot cuz eww
    ephemeris = np.array(ephemeris)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot zeta_g
    axes[0].plot(times, ephemeris[:, 0])
    axes[0].set_ylabel('ζ_g (m)')
    axes[0].grid(True)
    
    # Plot eta_g
    axes[1].plot(times, ephemeris[:, 1])
    axes[1].set_ylabel('η_g (m)')
    axes[1].grid(True)
    
    # Plot theta_g
    axes[2].plot(times, ephemeris[:, 2])
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('θ_g (rad)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()



    
