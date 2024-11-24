# bicycle.py - bicycle dynamics
# RMM, 24 Nov 2024 (from KJA)
#
# Bicyle dynamics
#
# This model describes the dynamics of a bicycle with the feature that one
# of its key properties is due to a feedback mechanism that is created by
# the design of the front fork.  This model is described in more detail in
# Section 4.2 of FBS2e.
#
# MATLAB header (bicycle_stabplot.m)
# % Linearized 4th order model and analysis of eigenvalues
# % Equations based on Schwab et al 2004
# % Run bicycleparameters first
# % kja 040611
# % Parameters of a bicycle model
# % kja 040613
# % Basic data is given by 26 parameters

import numpy as np
import control as ct
from math import pi

import numpy as np

# Acceleration of gravity [m/s^2]
g = 9.81

# Wheel base [m]
b = 1.00

# Trail [m]
c = 0.08

# Wheel radii
Rrw = 0.35
Rfw = 0.35

# Head angle [radians]
lambda_angle = np.pi * 70 / 180

# Rear frame mass [kg], center of mass [m], and inertia tensor [kgm^2]
mrf = 87
xrf = 0.491586
zrf = 1.028138
Jxxrf = 3.283666
Jxzrf = 0.602765
Jyyrf = 3.8795952
Jzzrf = 0.565929

# Front frame mass [kg], center of mass [m], and inertia tensor [kgm^2]
mff = 2
xff = 0.866
zff = 0.676
Jxxff = 0.08
Jxzff = -0.02
Jyyff = 0.07
Jzzff = 0.02

# Rear wheel mass [kg], center of mass [m], and inertia tensor [kgm^2]
mrw = 1.5
Jxxrw = 0.07
Jyyrw = 0.14

# Front wheel mass [kg], center of mass [m], and inertia tensor [kgm^2]
mfw = 1.5
Jxxfw = 0.07
Jyyfw = 0.14

# Auxiliary variables
xrw = 0
zrw = Rrw
xfw = b
zfw = Rfw
Jzzrw = Jxxrw
Jzzfw = Jxxfw

# Total mass
mt = mrf + mrw + mff + mfw

# Center of mass
xt = (mrf * xrf + mrw * xrw + mff * xff + mfw * xfw) / mt
zt = (mrf * zrf + mrw * zrw + mff * zff + mfw * zfw) / mt

# Inertia tensor components
Jxxt = (
    Jxxrf + mrf * zrf**2 +
    Jxxrw + mrw * zrw**2 +
    Jxxff + mff * zff**2 +
    Jxxfw + mfw * zfw**2
)
Jxzt = (
    Jxzrf + mrf * xrf * zrf +
    mrw * xrw * zrw +
    Jxzff + mff * xff * zff +
    mfw * xfw * zfw
)
Jzzt = (
    Jzzrf + mrf * xrf**2 +
    Jzzrw + mrw * xrw**2 +
    Jzzff + mff * xff**2 +
    Jzzfw + mfw * xfw**2
)

# Front frame parameters
mf = mff + mfw
xf = (mff * xff + mfw * xfw) / mf
zf = (mff * zff + mfw * zfw) / mf

Jxxf = (
    Jxxff + mff * (zff - zf)**2 +
    Jxxfw + mfw * (zfw - zf)**2
)
Jxzf = (
    Jxzff + mff * (xff - xf) * (zff - zf) +
    mfw * (xfw - xf) * (zfw - zf)
)
Jzzf = (
    Jzzff + mff * (xff - xf)**2 +
    Jzzfw + mfw * (xfw - xf)**2
)

# Auxiliary variables
d = (xf - b - c) * np.sin(lambda_angle) + zf * np.cos(lambda_angle)
Fll = (
    mf * d**2 +
    Jxxf * np.cos(lambda_angle)**2 +
    2 * Jxzf * np.sin(lambda_angle) * np.cos(lambda_angle) +
    Jzzf * np.sin(lambda_angle)**2
)
Flx = mf * d * zf + Jxxf * np.cos(lambda_angle) + Jxzf * np.sin(lambda_angle)
Flz = mf * d * xf + Jxzf * np.cos(lambda_angle) + Jzzf * np.sin(lambda_angle)
gamma = c * np.sin(lambda_angle) / b
Sr = Jyyrw / Rrw
Sf = Jyyfw / Rfw
St = Sr + Sf
Su = mf * d + gamma * mt * xt

# Matrices for the linearized fourth-order model
M = np.array([
    [Jxxt, -Flx - gamma * Jxzt],
    [-Flx - gamma * Jxzt, Fll + 2 * gamma * Flz + gamma**2 * Jzzt]
])

K0 = np.array([
    [-mt * g * zt, g * Su],
    [g * Su, -g * Su * np.cos(lambda_angle)]
])

K2 = np.array([
    [0, -(St + mt * zt) * np.sin(lambda_angle) / b],
    [0, (Su + Sf * np.cos(lambda_angle)) * np.sin(lambda_angle) / b]
])

c12 = gamma * St + Sf * np.sin(lambda_angle) \
    + Jxzt * np.sin(lambda_angle) / b + gamma * mt * zt
c22 = Flz * np.sin(lambda_angle) / b \
    + gamma * (Su + Jzzt * np.sin(lambda_angle) / b)

C = np.array([
    [0, -c12],
    [gamma * St + Sf * np.sin(lambda_angle), c22]
])

def whipple_A(v0):
    return np.block([
        [np.zeros((2, 2)), np.eye(2)],
        [-np.linalg.inv(M) @ (K0 + K2 * v0**2), -np.linalg.inv(M) @ C * v0]
    ])
