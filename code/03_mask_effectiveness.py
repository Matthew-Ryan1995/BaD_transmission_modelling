#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:17:30 2023

@author: rya200
"""
# %% libraries
import numpy as np
from msir import msir
import scipy as spi
import matplotlib.pyplot as plt
import os
# import time

working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %% Set up
TS = 1.0
ND = 1000.0

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)

# Inital conditions for incestion
# S0_m = 0.
# I0_m = 0
# I0_n = 1e-2  # 1% start infected
# R0_n = 0
# R0_m = 0
# S0_n = 1 - S0_m - I0_m - I0_n - R0_n - R0_m
# init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n)

# Enter custom params
cust_params = dict()
# cust_params["transmission"] = 1.5
# cust_params["infectious_period"] = 7
# cust_params["immune_period"] = 180
# # cust_params["av_lifespan"] = 0
# cust_params["susc_mask_efficacy"] = 0
# cust_params["inf_mask_efficacy"] = 0
# cust_params["nomask_social"] = 0
# cust_params["nomask_fear"] = 0
cust_params["mask_social"] = 10
# cust_params["mask_fear"] = 0

model = msir(**cust_params)

TS = 1.0
ND = 100.0

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)

# Inital conditions
# Note the order of conditions (M-N)
S0_m = 0.
I0_m = 0
I0_n = 1e-2  # 1% start infected
R0_n = 0
R0_m = 0
S0_n = 1 - S0_m - I0_m - I0_n - R0_n - R0_m
init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n)

# %% Mask parameter set up

p = c = np.linspace(0, 1, 20)

x, y = np.meshgrid(p, c)

params = np.array([x, y]).reshape(2, -1).T

# %% Run some


def get_inf(p, c):

    model.susc_mask_efficacy = c
    model.inf_mask_efficacy = p

    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=init_cond,
                                  # t_eval=t_range
                                  )
    dat = RES.y.T

    Im = dat[:, 2].max()
    In = dat[:, 3].max()

    return np.array([Im, In])


ANS = np.vstack(list(map(get_inf, params[:, 0], params[:, 1])))

# %% Figure
plt.figure()
plt.tripcolor(params[:, 0], params[:, 1], ANS[:, 0],  cmap="RdBu_r")
plt.title("Maximum proportion infected w/mask")
plt.xlabel("Efficacy to reduce spead (p)")
plt.ylabel("Efficacy to reduce contraction risk (c)")
plt.colorbar()
plt.show()

plt.figure()
plt.tripcolor(params[:, 0], params[:, 1], ANS[:, 1], cmap="RdBu_r")
plt.title("Maximum proportion infected w/no mask")
plt.xlabel("Efficacy to reduce spead (p)")
plt.ylabel("Efficacy to reduce contraction risk (c)")
plt.colorbar()
plt.show()

plt.figure()
plt.tripcolor(params[:, 0], params[:, 1], ANS[:, 0] + ANS[:, 1], cmap="RdBu_r")
plt.title("Maximum proportion infected")
plt.xlabel("Efficacy to reduce spead (p)")
plt.ylabel("Efficacy to reduce contraction risk (c)")
plt.colorbar()
plt.show()
