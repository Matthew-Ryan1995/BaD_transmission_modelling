#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:43:24 2023

@author: rya200
"""
# %%

from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

# %%

heat_map_params = dict()
heat_map_params["transmission"] = 1
heat_map_params["infectious_period"] = 1/1
heat_map_params["immune_period"] = 1/0.5
heat_map_params["av_lifespan"] = 0  # Turning off demography
heat_map_params["susc_B_efficacy"] = 0.5
heat_map_params["inf_B_efficacy"] = 0.3
heat_map_params["N_social"] = 0.2
heat_map_params["N_fear"] = 1.1
heat_map_params["B_social"] = 1.3
heat_map_params["B_fear"] = 0.1
heat_map_params["B_const"] = 0.
heat_map_params["N_const"] = 0.9

r0_a = np.arange(0.1, 5, step=0.1)
r0_a_range = np.arange(0.1, 5, step=0.1)
r0_b_range = np.arange(0.1, 3, 0.01)

beta_vals = list()
M3 = bad(**heat_map_params)
for idx in range(len(r0_a)):

    w = r0_a[idx]
    ww = w * (heat_map_params["N_social"] +
              heat_map_params["N_fear"] + heat_map_params["N_const"])

    M3.update_params(**{"B_social": ww})

    tmp = M3.Rzero()/heat_map_params["transmission"]

    beta_vals.append(1/tmp)

r0_d = np.array(beta_vals) * heat_map_params["infectious_period"]

grid_range = np.meshgrid(r0_a_range, r0_b_range)

iter_vals = np.array(grid_range).reshape(2, len(r0_a_range)*len(r0_b_range)).T
col_vals = list()
for idxx in range(len(iter_vals)):
    a1 = next(i for i in range(len(r0_a)) if iter_vals[idxx, 0] <= r0_a[i])

    b1 = r0_d[a1]

    if (iter_vals[idxx, 1] < b1) and (iter_vals[idxx, 0] < 1):
        col_vals.append(0)
    elif (iter_vals[idxx, 1] < b1):
        col_vals.append(1)
    else:
        col_vals.append(2)

# %%

plt.figure()
plt.plot([1, 1], [0, 3], ":k")
plt.plot(r0_a, r0_d, "r")
plt.contourf(grid_range[0], grid_range[1], np.array(
    col_vals).reshape(grid_range[0].shape))
plt.plot([0, 5], [1, 1], ":k")
# plt.ylim([0, max(r0_d) + 0.5])
plt.show()

# %%
tmp_dict = dict(heat_map_params)

tmp_dict["transmission"] = 1.01
tmp_dict["B_social"] = 0

R0_a = tmp_dict["B_social"] / \
    (tmp_dict["N_const"] + tmp_dict["N_fear"] + tmp_dict["N_social"])
R0_d = tmp_dict["transmission"] * tmp_dict["infectious_period"]

print(f"R0a = {R0_a}, R0d = {R0_d}")


ss, _ = find_ss(tmp_dict)

print(ss)

# %%

# R0_a + tmp_dict["B_fear"] * (tmp_dict["transmission"] * (1 - tmp_dict["inf_B_efficacy"]) + 1/tmp_dict["infectious_period"]) /((tmp_dict["N_const"] + tmp_dict["N_fear"] + tmp_dict["N_social"])/tmp_dict["infectious_period"])
# R0_a + tmp_dict["B_fear"] * ( 1/tmp_dict["infectious_period"]) /((tmp_dict["N_const"] + tmp_dict["N_fear"] + tmp_dict["N_social"])/tmp_dict["infectious_period"])
