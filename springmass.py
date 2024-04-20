# springmass.py - Spring mass dynamics
# RMM, 19 Apr 2024

import control as ct

m, k, b = 250, 40, 60
A = [[0, 1], [-k/m, -b/m]]
B = [[0], [1/m]]
C = [[1, 0]]

springmass = ct.ss(A, B, C, 0)
