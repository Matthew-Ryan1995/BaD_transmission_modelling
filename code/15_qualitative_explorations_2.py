#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:44:52 2023

@author: rya200
"""
import scipy.ndimage
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

# %%

P = 1
Ib0, Rb0, Rn0 = np.zeros(3)
Sb0 = 1e-3  # 1 in a million seeded with behaviour
In0 = 1e-3  # 1 in a million seeded with disease
# Ib0, Rb0, Rn0 = np.zeros(3)
# Sb0 = 1-0.6951793156273507  # 1 in a million seeded with behaviour
# In0 = Ib0 = 1e-6  # 1 in a million seeded with disease

Sn0 = P - Sb0 - Ib0 - Rb0 - In0 - Rn0

PP = np.array([Sn0, Sb0, In0, Ib0, Rn0, Rb0])

R0 = 1.5
gamma = 1
w1 = 2 * gamma
p = 0.6
c = 0.3

cust_params = dict()
cust_params["transmission"] = R0*gamma
cust_params["infectious_period"] = 1/gamma
cust_params["immune_period"] = 0.5
cust_params["av_lifespan"] = 0  # Turning off demography
cust_params["susc_B_efficacy"] = 0.
cust_params["inf_B_efficacy"] = 0.
cust_params["N_social"] = 0.8
cust_params["B_social"] = 0.05 * w1
cust_params["B_fear"] = w1
cust_params["B_const"] = 0.01
cust_params["N_const"] = 0.01
# w1 = 8
# R0 = 5
# gamma = 0.4
# w1 = 5 * gamma
# p = 0.6
# c = 0.3

# cust_params = dict()
# cust_params["transmission"] = R0*gamma
# cust_params["infectious_period"] = 1/gamma
# cust_params["immune_period"] = 240
# cust_params["av_lifespan"] = 0  # Turning off demography
# cust_params["susc_B_efficacy"] = 0.
# cust_params["inf_B_efficacy"] = 0.
# cust_params["N_social"] = 0.5
# cust_params["B_social"] = 0.05 * w1
# cust_params["B_fear"] = w1
# cust_params["B_const"] = 0.01
# cust_params["N_const"] = 0.01

M = bad(**cust_params)
M.run(PP, 0, 600)


plt.figure()
plt.plot(M.t_range, M.get_I(), "red")

M.update_params(**{"susc_B_efficacy": c, "inf_B_efficacy": p})
M.run(PP, 0, 600)
plt.plot(M.t_range, M.get_I(), "blue")

plt.show()
