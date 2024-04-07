# predprey_ctstime.py - Predator-prey model in continuous time
# RMM, 28 May 2023

import matplotlib.pyplot as plt
import numpy as np
import control as ct
import fbs                      # FBS plotting customizations

# Define the dynamis for the predator-prey system (no input)
def predprey_update(t, x, u, params={}):
    """Predator prey dynamics"""
    r = params.get('r', 1.6)
    d = params.get('d', 0.56)
    b = params.get('b', 0.6)
    k = params.get('k', 125)
    a = params.get('a', 3.2)
    c = params.get('c', 50)

    # Dynamics for the system
    dx0 = r * x[0] * (1 - x[0]/k) - a * x[1] * x[0]/(c + x[0])
    dx1 = b * a * x[1] * x[0] / (c + x[0]) - d * x[1]

    return np.array([dx0, dx1])

# Create a nonlinear I/O system
predprey_sys = ct.nlsys(predprey_update, states=2)

# Simulate a trajectory leading to a limit cycle
timepts = np.linspace(0, 70, 500)
sim = ct.input_output_response(predprey_sys, timepts, 0, [25, 20])

# Plot the results
fbs.figure('mlh')                                       # FBS  conventions
plt.plot(sim.time, sim.states[0], 'b-', label="Hare")
plt.plot(sim.time, sim.states[1], 'r--', label="Lynx")
plt.legend()
plt.xlabel("Time $t$ [years]")
plt.ylabel("Population")
plt.title("Time response")

# Save the figure
fbs.savefig('figure-4.20-predprey_ctstime-sim.png')     # PNG for web

# Generate a phase portrait
fbs.figure('mlh')
ct.phaseplot.equilpoints(predprey_sys, [-5, 126, -5, 100])
ct.phaseplot.streamlines(
    predprey_sys, np.array([
        [0, 100], [1, 0],
    ]), 10, color='b')
ct.phaseplot.streamlines(
    predprey_sys, np.array([[124, 1]]), np.linspace(0, 10, 500), color='b')
ct.phaseplot.streamlines(
    predprey_sys, np.array([[125, 25], [125, 50], [125, 75]]), 3, color='b')
ct.phaseplot.streamlines(predprey_sys, np.array([2, 8]), 6, color='b')
ct.phaseplot.streamlines(
    predprey_sys, np.array([[20, 30]]), np.linspace(0, 65, 500),
    gridtype='circlegrid', gridspec=[2, 1], arrows=10, color='r')
ct.phaseplot.vectorfield(predprey_sys, [5, 125, 5, 100], gridspec=[20, 20])

# Add the limit cycle
resp1 = ct.initial_response(predprey_sys, np.linspace(0, 100), [20, 75])
resp2 = ct.initial_response(
    predprey_sys, np.linspace(0, 20, 500), resp1.states[:, -1])
plt.plot(resp2.states[0], resp2.states[1], color='k')

# Legacy code
# def pp_ode(x, t):
#     return predprey_update(t, x, 0, {})
# ct.phase_plot(pp_ode, [0, 60, 7], [0, 50, 6])
# ct.phase_plot(pp_ode, [0, 60, 7], [60, 100, 4])
# ct.phase_plot(pp_ode, [70, 120, 6], [0, 50, 6])
# ct.phase_plot(pp_ode, [70, 120, 6], [60, 100, 4])

# # Plot the limit cycle
# ct.phase_plot(pp_ode, X0=sim.states[:, -1:].T, T=20)    # limit cycle
# ct.phase_plot(pp_ode, X0=[[120, 32], [120, 60]], T=20)  # outside trajectories
# ct.phase_plot(pp_ode, X0=[[19, 30]], T=75)              # inside trajectories

# Label the plot
plt.xlabel("Hares")
plt.ylabel("Lynxes")
plt.title("Phase portrait")
fbs.savefig('figure-4.20-predprey_ctstime-pp.png')      # PNG for web
