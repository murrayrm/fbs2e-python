# predprey_place.py - stabilization via state space feedback
# RMM, 18 Jan 2026

import matplotlib.pyplot as plt
import numpy as np
import control as ct
import fbs                      # FBS plotting customizations

# System definition
from predprey import predprey

# Find the equilibrium point and linearize
xe, ue = ct.find_eqpt(predprey, [20, 30], 0)
sys = predprey.linearize(xe, ue)

# Eigenvalue placement
K = ct.place(sys.A, sys.B, [-0.1, -0.2])
kf = -1 / (sys.C[1] @ np.linalg.inv(sys.A - sys.B @ K) @ sys.B)

ctrl = ct.nlsys(
    None, lambda t, x, u, params: -K @ (u[0:2] - xe) + kf * (u[2] - xe[1]),
    inputs=['H', 'L', 'r'], outputs=['u'],
)
clsys = ct.interconnect(
    [predprey, ctrl], inputs=['r'], outputs=['H', 'L', 'u'],
    name='predprey_clsys'
)

# Compute initial condition response
T = np.linspace(0, 100, 1000)
response = ct.input_output_response(clsys, T, 30, [20, 15])

# Plot results
fbs.figure('mlh')               # Create figure using FBS defaults
plt.plot(response.time, response.outputs[0], 'b-', label="Hare")
plt.plot(response.time, response.outputs[1], 'r--', label="Lynx")

plt.xlabel("Time [years]")
plt.ylabel("Population")
plt.legend(frameon=False)

plt.tight_layout()
fbs.savefig('figure-7.7-predprey_place-time.png')

# Generate a phase portrait
fbs.figure('mlh')               # Create figure using FBS defaults

# Create a function for the phase portrait that includes the reference input
ppfcn = lambda t, x: clsys.dynamics(t, x, [30], {})

ct.phase_plane_plot(
    ppfcn, [1, 100, 1, 100], 30, plot_separatrices=False)
# Turn off vector field lines since phase_plan_plot has arrows
# ct.phaseplot.vectorfield(ppfcn, [0, 100, 0, 100], gridspec=[20, 20])

plt.xlabel("Hares")
plt.ylabel("Lynxes")
plt.suptitle("")

plt.tight_layout()
fbs.savefig('figure-7.7-predprey_place-pp.png')
