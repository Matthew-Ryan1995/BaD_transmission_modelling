#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:50:18 2023

@author: rya200
"""
# %% Packages/libraries
import math
from scipy.integrate import quad
from scipy.optimize import fsolve
import numpy as np
import scipy as spi
import matplotlib.pyplot as plt
import json
import os
import time
from msir_4 import *

working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %%

TS = 1.0/14
ND = 600.0

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)

# Inital conditions
N = 1
# Note the order of conditions (M-N)
S0_m = 1e-6
I0_m = 0
I0_n = 1e-3  # 1% start infected
R0_n = 0
R0_m = 0
S0_n = N - S0_m - I0_m - I0_n - R0_n - R0_m
FS = 0
FS_m = 0
init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n, FS, FS_m)

w1 = 8
R0 = 0.9
gamma = 0.4
# Enter custom params
# cust_params = dict()
# cust_params["transmission"] = R0*0.1
# cust_params["infectious_period"] = 1/0.1
# # cust_params["immune_period"] = 240
# cust_params["av_lifespan"] = 0
# cust_params["susc_mask_efficacy"] = 0.4
# cust_params["inf_mask_efficacy"] = 0.8
# cust_params["nomask_social"] = 0.
# cust_params["nomask_fear"] = 0.0
# cust_params["mask_social"] = w1  # 0.0 * w1
# cust_params["mask_fear"] = 0  # w1
# cust_params["mask_const"] = 0.
# cust_params["nomask_const"] = 0.01
# model = msir(**cust_params)
cust_params = dict()
cust_params["transmission"] = R0*gamma
cust_params["infectious_period"] = 1/gamma
cust_params["immune_period"] = 240
cust_params["av_lifespan"] = 0
cust_params["susc_mask_efficacy"] = 0.8
cust_params["inf_mask_efficacy"] = 0.4
cust_params["nomask_social"] = 0.7
cust_params["nomask_fear"] = 0.0
cust_params["mask_social"] = 0.6
cust_params["mask_fear"] = 0.2
cust_params["mask_const"] = 0.1
cust_params["nomask_const"] = 0.1
model = msir(**cust_params)

RES = spi.integrate.solve_ivp(fun=model.run,
                              t_span=[t_start, t_end],
                              y0=init_cond,
                              t_eval=t_range,
                              # events=[event]
                              )
dat = RES.y.T

Istar = dat[-1, 2:4].sum()

if model.mask_social - model.nomask_social != 0:
    C = model.mask_social - model.nomask_social
    D = model.mask_const + model.mask_fear*Istar + \
        model.nomask_fear*(1-Istar) + model.nomask_const
    Delta = ((C + D) - np.sqrt((C + D)**2 - 4*C *
                               (D - (model.mask_const + model.mask_fear*Istar))))/(2*C)
    Delta2 = ((C + D) + np.sqrt((C + D)**2 - 4*C *
                                (D - (model.mask_const + model.mask_fear*Istar))))/(2*C)
elif (model.mask_const + model.nomask_fear + model.nomask_const) == 0:
    Delta = S0_m/N
else:
    Delta = (model.nomask_fear + model.nomask_const) / \
        (model.mask_const + model.nomask_fear + model.nomask_const)

plt.figure()
plt.plot((dat[:, 0] + dat[:, 2] + dat[:, 4])/N, label="Masks")
plt.plot((dat[:, 1] + dat[:, 3] + dat[:, 5])/N, label="No Masks")

plt.plot([0, dat[:, 1].size], [Delta, Delta], ":k")
plt.legend()
plt.xlabel("time")
plt.ylabel("proportion")
plt.show()
plt.figure()

plt.plot((dat[:, 0] + dat[:, 1])/N, label="S")
plt.plot((dat[:, 2] + dat[:, 3])/N, label="I")
plt.plot((dat[:, 4] + dat[:, 5])/N, label="R")
plt.legend()
plt.xlabel("time")
plt.ylabel("proportion")
plt.show()

print(f"Est delta is {Delta}")
print(f"Est delta is {Delta2}")
# print(
# f"Est mathsy is {model.nomask_const/(model.mask_social-model.nomask_social)}")
