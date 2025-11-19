'''tests the dynamics'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import src.uav_dynamics as uav_dynamics


def test_nl_prop():

    init_state = [0,0,0]
    control_vector = [1, 0]
    step = 1

    uav = uav_dynamics.Dynamical_UAV(init_state)
    uav.step_nl_propagation(control_vector, step)

    print(uav.current_state)


    
