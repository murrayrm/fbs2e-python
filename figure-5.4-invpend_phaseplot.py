# invepend_phaseplot.py - inverted pendulum phase plots
# RMM, 6 Apr 2024

import matplotlib.pyplot as plt
import numpy as np
from math import pi
import control as ct
import fbs                      # FBS plotting customizations

def invpend_update(t, x, u, params):
    m, l, b, g = params['m'], params['l'], params['b'], params['g']
    return [x[1], -b/m * x[1] + (g * l / m) * np.sin(x[0])]
invpend = ct.nlsys(
    invpend_update, states=2, inputs=0, name='inverted pendulum',
    params={'m': 1, 'l': 1, 'b': 0.2, 'g': 1})

fbs.figure()
ct.phase_plane_plot(
    invpend, [-2*pi, 2*pi, -2, 2], 4, gridspec=[6, 6],
    plot_separatrices={'timedata': 20, 'arrows': 4})
fbs.savefig('figure-5.4-invpend_phaseplot.png')
