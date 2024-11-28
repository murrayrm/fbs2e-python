# example-6.8-opamp_bandpass.py - <Short description>
# RMM, 28 Nov 2028
#
# Figure 6.12: Active band-pass filter. The circuit diagram (a) shows an op
# amp with two RC filters arranged to provide a band-pass filter.  The plot
# in (b) shows the gain and phase of the filter as a function of frequency.
# Note that the phase starts at -90 deg due to the negative gain of the
# operational amplifier.

import control as ct
import numpy as np
import matplotlib.pyplot as plt
ct.use_fbs_defaults()

#
# System dynamics
#

# Filter parameters
R1 = 100
R2 = 5000
C1 = 100e-6
C2 = 100e-6

# State-space form of the solution
A = np.array([[-1 / (R1 * C1), 0],
              [1 / (R1 * C2), -1 / (R2 * C2)]])
B = np.array([[1 / (R1 * C1)],
              [-1 / (R1 * C2)]])
C = np.array([[0, 1]])
D = np.array([[0]])

# Create the state-space system
sys = ct.ss(A, B, C, D)

#
# (b) Frequency response
#

# Generate the Bode plot data
omega = np.logspace(-1, 3, 100)  # Frequency range from 0.1 to 1000 rad/s
cplt = ct.frequency_response(sys, omega).plot(
    initial_phase=-90, title="(b) Frequency response")

# Plot the unit gain line
cplt.axes[0, 0].axhline(1, color='k', linestyle='-', linewidth=0.5)

# Print system poles for bandwidth computation
print("System poles (for computing bandwidth):")
print(sys.poles())

# Save the figure
plt.savefig("figure-6.8-opamp_bandpass.png", bbox_inches='tight')
