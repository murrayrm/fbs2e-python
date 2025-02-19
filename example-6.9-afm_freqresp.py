# example-6.9-afm_freqresp.py - <Short description>
# RMM, 28 Nov 2024
#
# Figure 6.13: AFM frequency response. (a) A block diagram for the vertical
# dynamics of an atomic force microscope in contact mode. The plot in (b)
# shows the gain and phase for the piezo stack. The response contains two
# frequency peaks at resonances of the system, along with an antiresonance
# at ω = 268 krad/s.  The combination of a resonant peak followed by an
# antiresonance is common for systems with multiple lightly damped
# modes. The dashed horizontal line represents the gain equal to the zero
# frequency gain divided by sqrt(2).

import control as ct
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
ct.use_fbs_defaults()

#
# System dynamics
#

# System parameters
m1 = 0.15e-3
m2 = 1e-3
f1 = 40.9e3
f2 = 41.6e3
f3 = 120e3

# Derived quantities
w1 = 2 * np.pi * f1
w2 = 2 * np.pi * f2
w3 = 2 * np.pi * f3
z1 = 0.1
z3 = 0.1
k2 = w2**2 * m2
c2 = m2 * z1 * w1

# System matrices
A = np.array([[0, 1, 0, 0],
              [-k2 / (m1 + m2), -c2 / (m1 + m2), 1 / m2, 0],
              [0, 0, 0, 1],
              [0, 0, -w3**2, -2 * z3 * w3]])
B = np.array([[0], [0], [0], [w3**2]])
C = (m2 / (m1 + m2)) * np.array([m1 * k2 / (m1 + m2), m1 * c2 / (m1 + m2), 1, 0])
D = np.array([[0]])

# Create the state-space system
sys = ct.ss(A, B, C, D)

#
# (b) Frequency response
#

# Generate the frequency response
omega = np.logspace(4, 7, 10000)  # Frequency range from 10^4 to 10^7 rad/s
freqresp = ct.frequency_response(sys, omega)
mag, phase = freqresp.magnitude, freqresp.phase

# Create the Bode plot
cplt = freqresp.plot()
mag_ax, phase_ax = cplt.axes[:, 0]
cplt.set_plot_title("(b) Frequency response")

# Locate peaks and valleys
# Find the first peak
max_mag = 0
omega1 = None
for i, m in enumerate(mag):
    if m > max_mag:
        max_mag = m
        omega1 = omega[i]
    elif m < max_mag:
        break

# Find the first valley
min_mag = max_mag
omega2 = None
for i, m in enumerate(mag):
    if m < min_mag:
        min_mag = m
        omega2 = omega[i]
    elif m > min_mag:
        break

# Find the second peak (must be higher than first)
omega3 = None
for i, m in enumerate(mag):
    if m > max_mag:
        max_mag = m
        omega3 = omega[i]
    elif m < max_mag and omega3 is not None:
        break

# Print peaks and valley frequencies
print(f"Peaks at {omega1:.2e}, {omega3:.2e}; valley at {omega2:.2e}")

# Add lines to mark the frequencies
mag_ax.axhline(1 / sqrt(2), color='r', linestyle='--', linewidth=0.5)

mag_ax.axvline(omega1, color='k', linestyle='--', linewidth=0.5)
mag_ax.text(omega1 * 1.1, 2, r"$M_{r1}$")
mag_ax.text(omega1 * 1.1, 0.07, r"$\omega = \omega_{r1}$")

mag_ax.axvline(omega3, color='k', linestyle='--', linewidth=0.5)
mag_ax.text(omega3 * 1.2, 3, r"$M_{r2}$")
mag_ax.text(omega3 * 1.1, 0.07, r"$\omega = \omega_{r2}$")

phase_ax.axvline(omega1, color='k', linestyle='--', linewidth=0.5)
phase_ax.axvline(omega3, color='k', linestyle='--', linewidth=0.5)

# Save the figure
plt.savefig("figure-6.9-afm_freqresp.png", bbox_inches='tight')
