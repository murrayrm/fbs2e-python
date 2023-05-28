# congctrl.py - Congestion control dynamics and control
# RMM, 11 Jul 2006 (from MATLAB)
#
# Congestion control dynamics
#
# This model implements the congestion control dynamics described in
# the text.  We assume that we have N identical sources and 1 router.
# 
# To allow the model to be used in a variety of ways, we
# allow the number of states and the number of sources to be set
# independently.  The length of the state vector, M+1, is used to determine
# the number of simulated sources, with each source representing N/M
# sources.

import numpy as np
import control as ct

# Congestion control dynamics
def _congctrl_update(t, x, u, params):
    # Number of sources per state of the simulation
    M = x.size - 1

    # Remaining parameters
    N = params.get('N', M)              # number of sources
    rho = params.get('rho', 2e-4)       # RED parameter = pbar / (bupper-blower)
    c = params.get('c', 10)             # link capacity (Mp/ms)

    # Compute the derivative (last state = bdot)
    return np.append(
        c / x[M] - (rho * c) * (1 + (x[:-1]**2) / 2),
        N/M * np.sum(x[:-1]) * c / x[M] - c)


# Function to define an I/O system
def create_iosystem(M, N=60, rho=2e-4, c=10):
    #! TODO: Check to make sure M and N are compatible
    return ct.NonlinearIOSystem(
        _congctrl_update, None, states=M+1,
        params={'N': N, 'rho': rho, 'c': c})
