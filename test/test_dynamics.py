'''tests the dynamics'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import src.dynamics as dynamics



def test_propagation():

    start = dynamics.StateVector(0, 0, 0)

    ugv = dynamics.UGV(start)
    t_evals = np.arange(0,10,0.1)
    control = np.array([0.1, 0.1])
    sol = ugv.propagate(t_evals, control)


    
