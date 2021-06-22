# figure-2.11-2dof_stepresp.py - step responses for two DOF system
# RMM, 21 Jun 2021
#
# Response to a step change in the reference signal for a system with a PI
# controller having two degrees of freedom. The process transfer function is
# P(s) = 1/s and the controller gains are kp = 1.414, ki = 1, and β = 0,
# 0.5, and 1.
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import control as ct

s = ct.TransferFunction.s           # define the differentiation operator
mpl.rcParams['text.usetex'] = True  # use LaTeX for formatting legend strings

# Process model: integrator dynamics
P = ct.tf([1], [1, 0])

# Set the simulation time vector
time = np.linspace(0, 10, 100)

#
# Beta sweep
#

# Choose gains to use
beta_list = [0, 0.5, 1]
kp = 1.414
ki = 1

for beta in beta_list:
    C1 = beta * kp + ki / s
    C2 = (1 - beta) * kp

    Gyr = P * C1 / (1 + P * (C1 + C2))
    Gur = C1 / (1 + P * (C1 + C2))

    t, y = ct.step_response(Gyr, time)
    t, u = ct.step_response(Gur, time)

    if 'w_y_ax' not in locals():
        w_y_ax = plt.subplot(3, 2, 1)
        plt.plot([time[0], time[-1]], [1, 1], 'k-')
        plt.xlabel('Time $t$')
        plt.ylabel('Output $y$')
    w_y_ax.plot(t, y)

    if 'w_u_ax' not in locals():
        w_u_ax = plt.subplot(3, 2, 2)
        plt.xlabel('Time $t$')
        plt.ylabel('Input $u$')
    w_u_ax.plot(t, u, label=r"$\beta = %g$" % beta)

# Label the omega sweep curves
w_u_ax.legend(loc="upper right")

# Overalll figure formating
plt.tight_layout()
