# congctrl_eqplot.py - congestion control equilibrium point plot
# RMM, 29 Jun 2007 (converted from MATLAB)
#
# The equilibrium buffer size be for a set of N identical computers sending
# packets through a single router with drop probability ρb.
#

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import fbs                      # FBS plotting customizations

# Range of values to plot (\alpha = 1/(2\rho^2 N^2)
alpha_vals = np.logspace(-2, 4)

# Solve for the equilibrium value of \rho b_e
bratio_vals = []
for alpha in alpha_vals:
    # Define a function for the equilibrium point (equation (4.22))
    def equilibrium(bratio):
        return alpha * bratio**3 + bratio - 1

    bratio = scipy.optimize.fsolve(equilibrium, 0)
    bratio_vals.append(bratio)

# Set up a figure for plotting the results
fbs.figure('mlh')

# Plot the equilibrium buffer length
plt.semilogx(alpha_vals, bratio_vals)
plt.xlabel(r"$1/(2 \rho^2 N^2)$ (log scale)")
plt.ylabel(r"$\rho b_{e}$")
plt.title("Operating point")

# Save the figure
fbs.savefig('figure-4.12-congctrl_eqplot.png')          # PNG for web
