# example-3.15-queuing_systems.py - Queuing system modeling
# RMM, 29 Aug 2021
#
# Figure 3.15: <caption>

import control as ct
import numpy as np
import matplotlib.pyplot as plt

#
# System dynamics
#

# Set up the plotting grid to match the layout in the book
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)

#
# (a) Description
#

fig.add_subplot(gs[0, 0])       # first row, first column

#
# (b) Description
#

fig.add_subplot(gs[0, 1])       # first row, second column

# Save the figure
plt.savefig("figure-3.22-queuing_dynamics.png", bbox_inches='tight')
