# example-N.mm-short_title.py - <Short description>
# RMM, dd MMM yyyy
#
# Figure 3.15: <caption>

import control as ct
import numpy as np
import matplotlib.pyplot as plt
ct.use_fbs_defaults()

#
# System dynamics
#

# Set up the plotting grid to match the layout in the book
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)

#
# (a) Description
#

ax = fig.add_subplot(gs[0, 0])  # first row, first column
ax.set_title("(a) Description")

#
# (b) Description
#

ax = fig.add_subplot(gs[0, 1])  # first row, second column
ax.set_title("(b) Description")

# Save the figure
plt.savefig("figure-N.mm-short_title.png", bbox_inches='tight')
