# predprey.py - predator-prey dynamics
# RMM, 21 Apr 2024
#
# Predator-prey dynamics
#
# This model implements the dynamics of a predator-prey system, as
# described in Section 2.7 of FBS.

import numpy as np
import control as ct

# Define the dynamis for the predator-prey system (no input)
predprey_params = {'r': 1.6, 'd': 0.56, 'b': 0.6, 'k': 125, 'a': 3.2, 'c': 50}
def predprey_update(t, x, u, params):
    """Predator prey dynamics"""
    r, d, b, k, a, c = map(params.get, ['r', 'd', 'b', 'k', 'a', 'c'])
    u = np.atleast_1d(u)        # Fix python-control bug
    u = np.clip(u, 0, 4*r)    # constraints used in FBS 2e
    # u = np.clip(u, -r, 3*r)     # constrain the input to keep r_eff >= 0

    # Dynamics for the system
    dH = (r + u[0]) * x[0] * (1 - x[0]/k) - a * x[1] * x[0]/(c + x[0])
    dL = b * a * x[1] * x[0] / (c + x[0]) - d * x[1]

    return np.array([dH, dL])

# Create a nonlinear I/O system
predprey = ct.nlsys(
    predprey_update, name='predprey', params=predprey_params,
    states=['H', 'L'], inputs='u', outputs=['H', 'L'])
