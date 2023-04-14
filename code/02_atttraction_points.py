#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:30:53 2023

@author: rya200
"""

# %% libraries
import numpy as np
from msir import msir
import scipy as spi
import matplotlib.pyplot as plt
import os
import time

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
cust_params["immune_period"] = 180
# # cust_params["av_lifespan"] = 0
# cust_params["susc_mask_efficacy"] = 0
# cust_params["inf_mask_efficacy"] = 0
# cust_params["nomask_social"] = 0
# cust_params["nomask_fear"] = 0
# cust_params["mask_social"] = 0
# cust_params["mask_fear"] = 0

model = msir(**cust_params)

# %% Initial conditions array


grid_vals = 10

s_n = np.linspace(0, 1, grid_vals)
i_m = np.linspace(0, 1, grid_vals)
i_n = np.linspace(0, 1, grid_vals)
r_m = np.linspace(0, 1, grid_vals)
r_n = np.linspace(0, 1, grid_vals)

S_n, I_m, I_n, R_m, R_n = np.meshgrid(s_n, i_m, i_n, r_n, r_m, indexing="ij")


int_conds = np.array([S_n, I_m, I_n, R_m, R_n]).reshape(5, -1).T


def sum_less_one(t):
    return int_conds[t, :].sum() <= 1


def no_inf(t):
    return int_conds[t, 2:3].sum() <= 0


def remove_masks(t):
    return int_conds[t, 0:5:2].sum() > 0


# Remove non-feasible
idx = list(filter(sum_less_one, range(grid_vals**5)))

int_conds = int_conds[idx, :]

# Remove no infections
idx = list(filter(no_inf, range(int_conds.shape[0])))

int_conds = np.delete(int_conds, idx, 0)


S_m = 1.0 - np.sum(int_conds, 1)


int_conds = np.hstack([S_m.reshape(S_m.size, 1), int_conds])


# Remove masks
idx = list(filter(remove_masks, range(int_conds.shape[0])))

int_conds = np.delete(int_conds, idx, 0)


def run_model_return_stable(int_cond):
    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=int_cond,
                                  # t_eval=t_range
                                  )
    dat = RES.y.T
    return dat[-1, :]


def vary_int_cond(x):
    t, int_cond = x
    return run_model_return_stable(int_cond)


ANS = list(map(vary_int_cond, enumerate(int_conds)))
ANS = np.vstack(ANS)


# %% plotting

def rand_jitter(arr):
    stdev = .01  # * (max(arr) - min(arr))
    return arr + np.random.randn(1) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

# plt.figure()
# jitter(int_conds[:, 0], ANS[:, 0], c="r", label="Sm")
# jitter(int_conds[:, 1], ANS[:, 1], c="b", label="Sn")
# jitter(int_conds[:, 2], ANS[:, 2], c="g", label="Im")
# jitter(int_conds[:, 3], ANS[:, 3], c="y", label="In")
# jitter(int_conds[:, 4], ANS[:, 4], c="orange", label="Rm")
# jitter(int_conds[:, 5], ANS[:, 5], c="black", label="Rn")
# plt.legend()
# plt.show()


plt.figure()
c_list = ["r", "b", "g", "y", "orange", "black"]
l_list = ["Sm", "Sn", "Im", "In", "Rm", "Rn"]
for i in range(int_conds.shape[1]):
    for j in range(6):
        jitter(int_conds[i, j], ANS[i, j], c=c_list[j], label=l_list[j])
plt.legend(l_list)
plt.xlabel("initial condition")
plt.ylabel("equalibrium state")
plt.show()
