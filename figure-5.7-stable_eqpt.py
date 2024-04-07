# stable_eqpt.py - plots for stable equlibrium point
# RMM, 6 Apr 2024

import matplotlib.pyplot as plt
import numpy as np
from math import pi
import control as ct
import control.phaseplot as pp
import fbs                      # FBS plotting customizations

m, b, k = 1, 0, 2
linsys = ct.ss([[0, 1], [-k/m, -b/m]], [[0], [1]], np.eye(2), 0)

# Draw the phase portrait
fbs.figure()
ct.phase_plane_plot(linsys, [-1, 1, -1, 1], 1, plot_streamlines=False)
pp.streamlines(
    linsys, np.array([[0.2, 0], [0.4, 0], [0.6, 0], [0.8, 0], [1, 0]]),
    4.5, arrows=6)
plt.gca().set_aspect('equal')
plt.suptitle("")

# Add some level sets
theta = np.linspace(0, 2*pi)
plt.plot(0.2 * np.sin(theta), 0.2 * np.cos(theta), 'r--')
plt.plot(0.3 * np.sin(theta), 0.3 * np.cos(theta), 'r--')

fbs.savefig('figure-5.7-stable_eqpt-pp.png')

fbs.figure('321')
plt.axis([0, 10, -2.5, 2.5])
timepts = np.linspace(0, 10)
response = ct.input_output_response(linsys, timepts, 0, [1, 0])
plt.plot(response.time, response.outputs[0], 'b', label="$x_1$")
plt.plot(response.time, response.outputs[1], 'r--', label="$x_2$")
plt.xlabel("Time $t$")
plt.ylabel("$x_1, x_2$")
plt.legend(loc='upper right', ncols=2, frameon=False)
fbs.savefig('figure-5.7-stable_eqpt-time.png')
