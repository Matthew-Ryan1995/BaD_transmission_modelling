#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:23:09 2023

@author: rya200
"""

from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

# %%

params = load_param_defaults()


R0 = 1.4
params["transmission"] = R0
params["susc_B_efficacy"] = 1
params["inf_B_efficacy"] = 1
# params["B_const"] = 0

w1 = np.arange(0, 10, step=0.1)

for ww in w1:
    params["B_social"] = ww
    tmp, _ = find_ss(params)
    B = tmp[[1, 3, 5]].sum()
    I = tmp[[2, 3]].sum()
    print(f"B = {B}, I = {I}")
    if np.isclose(I, 0.0):
        break

beta = params["transmission"]
gamma = 1/params["infectious_period"]
nu = 1/params["immune_period"]

print(f"no behaviour I={nu*(beta-gamma)/(beta*(gamma+nu))}")
