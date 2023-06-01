#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:47:34 2023

@author: rya200
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R0 = 5
gamma = 0.1
beta = R0 * gamma
nu = 0.01

p = 1
c = 0.1

D = 0.4


def DE(t, PP):
    Y = np.zeros(6)

    Y[0] = -beta*(PP[2] + (1-p) * PP[3]) * PP[0] + nu * PP[4]
    Y[1] = -beta*(1-c) * (PP[2] + (1-p) * PP[3]) * PP[1] + nu * PP[5]
    Y[2] = beta*(PP[2] + (1-p) * PP[3]) * PP[0] - gamma * PP[2]
    Y[3] = beta*(1-c) * (PP[2] + (1-p) * PP[3]) * PP[1] - gamma * PP[3]
    Y[4] = gamma * PP[2] - nu * PP[4]
    Y[5] = gamma * PP[3] - nu * PP[5]
    return Y


I1 = 1e-3
I2 = 1e-3
R1 = 0
R2 = 0
S1 = D - I1 - R1
S2 = 1-D - I2 - R2

IC = [S1, S2, I1, I2, R1, R2]

t_start, t_end = [0, 600]
tt = np.arange(t_start, t_end, 1)

res = solve_ivp(DE, [t_start, t_end], y0=IC, t_eval=tt)

dat = res.y.T

if c == 0:
    Sn_eq = D*gamma/(beta * (D*p-p+1))
    Sm_eq = gamma * (1-D) / (beta * (D * p - p + 1))
    In_eq = D * nu * (beta*D*p - beta*p + beta - gamma) / \
        (beta * (gamma + nu) * (D*p - p + 1))
    Im_eq = - nu * (D - 1) * (beta*D*p - beta*p + beta - gamma) / \
        (beta * (gamma + nu) * (D*p - p + 1))
elif p == 1:
    Sn_eq = gamma/beta
    Sm_eq = gamma*(1-D)/(-beta*D*c + beta*D + c*gamma)
    In_eq = nu*(beta*D - gamma)/(beta*(gamma + nu))
    Im_eq = nu*(D-1)*(c-1)*(beta*D - gamma) / \
        ((gamma + nu) * (-beta*D*c + beta*D + c*gamma))

plt.xkcd()
plt.figure()
plt.plot(dat[:, 0], "y", label="S1", alpha=0.25)
plt.plot(dat[:, 1], "y:", label="S2", alpha=0.25)
plt.plot(dat[:, 2], "g", label="I1", alpha=0.25)
plt.plot(dat[:, 3], "g:", label="I2", alpha=0.25)
plt.plot([t_start, t_end], [Sn_eq, Sn_eq], "y")
plt.plot([t_start, t_end], [Sm_eq, Sm_eq], "y:")
plt.plot([t_start, t_end], [In_eq, In_eq], "g")
plt.plot([t_start, t_end], [Im_eq, Im_eq], "g:")
plt.legend()
plt.show()
