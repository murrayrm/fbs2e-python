# figure_name.py - short description
# RMM, date

import matplotlib.pyplot as plt
import numpy as np
import control as ct
import fbs                      # FBS plotting customizations

# System definition

# Compute system response

# Plot results
fbs.figure('mlh')               # Create figure using FBS defaults

plt.tight_layout()
fbs.savefig('figure-N.m-figure_name-panel.png')
