# unstable_eqpt.py - plots for stable equlibrium point
# RMM, 7 Apr 2024

import matplotlib.pyplot as plt
import numpy as np
import control as ct
import fbs                      # FBS plotting customizations

saddle = ct.ss([[1, -3], [-3, 1]], [[0], [1]], np.eye(2), 0)

# Draw the phase portrait
fbs.figure()
ct.phase_plane_plot(
    saddle, [-1, 1, -1, 1], 0.4,
    gridtype='meshgrid', gridspec=[6, 6])
plt.gca().set_aspect('equal')
plt.suptitle("")
fbs.savefig('figure-5.9-unstable_eqpt-pp.png')

fbs.figure('321')
plt.axis([0, 3, -100, 100])
timepts = np.linspace(0, 3)
response = ct.input_output_response(saddle, timepts, 0, [1, 0])
plt.plot(response.time, response.outputs[0], 'b', label="$x_1$")
plt.plot(response.time, response.outputs[1], 'r--', label="$x_2$")
plt.xlabel("Time $t$")
plt.ylabel("$x_1, x_2$")
plt.legend(loc='upper right', ncols=1, frameon=False)
fbs.savefig('figure-5.9-unstable_eqpt-time.png')
