'''tests the dynamics'''

import numpy as np

from ..src import dynamics


def test_propagation():

    start = dynamics.JointStateVector(0, 0, 0)

    ugv = dynamics.UGV(start)
    t_evals = np.arange(0,10,0.1)
    control = np.array([0.1, 0.1])
    ugv.propagate(t_evals, control)
