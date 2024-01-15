#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:55:11 2023

Rough working, is S close enough to 1?

What are reasonable values for w1-3 and a1-3?

@author: rya200
"""
# %%
from msir_4 import *
import numpy as np
from scipy.integrate import solve_ivp
import json
import os
import time
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from multiprocessing import pool
from multiprocessing import cpu_count

working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %% Set up

tic = time.time()

TS = 1.0
ND = 300.0

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)

# Inital conditions
N = 1
# Note the order of conditions (M-N)
S0_m = 0.
I0_m = 0
I0_n = 1e-3  # 0.1% start infected
R0_n = 0
R0_m = 0
S0_n = N - S0_m - I0_m - I0_n - R0_n - R0_m
FS = 0
init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n, FS, 0)

# %%

R0 = 5
gamma = 0.1
beta = gamma*R0

p = c = 0.8


w2 = np.arange(0, 10, step=1)
w1 = 0.05*w2  # np.arange(0, 1, step=0.1)
w3 = np.arange(0, 1, step=0.1)
a2 = 1/(np.arange(0, 10, step=1) + 1)
a1 = np.arange(0, 1, step=0.1)
a3 = np.arange(0, 1, step=0.1)

# w1 = w1[int((len(w1) - 1)/2)]
# # w2 = w2[int((len(w2) - 1)/2)]
# w3 = w3[int((len(w3) - 1)/2)]
# a1 = a1[int((len(a1) - 1)/2)]
# a2 = a2[int((len(a2) - 1)/2)]
# a3 = a3[int((len(a3) - 1)/2)]


def event(t, y):
    if t > 10:
        ans = y[2] + y[3] - 1e-3
    else:
        ans = 1
    return ans


event.terminal = True


def get_final_size(idx):
    new_params = dict()

    new_params["transmission"] = beta
    new_params["susc_mask_efficacy"] = c
    new_params["inf_mask_efficacy"] = p
    new_params["mask_social"] = w1[idx]
    new_params["mask_fear"] = w2[idx]
    new_params["infectious_period"] = 1/gamma
    new_params["nomask_social"] = a1[idx]
    new_params["nomask_fear"] = a2[idx]
    new_params["nomask_const"] = a3[idx]
    new_params["mask_const"] = w3[idx]

    model = msir(**new_params)

    RES = solve_ivp(fun=model.run,
                    t_span=[t_start, t_end],
                    y0=init_cond,
                    t_eval=t_range,
                    events=[event]
                    )
    dat = RES.y.T

    return dat[-1, 6]


def sir_FS(x):
    init = S0_n/N
    return x - (1 - init * np.exp(-R0 * x))
# def sir_FS(x):
#     init = S

#     return np.log(init/x) - R0 * (1 - x/N)


if __name__ == "__main__":
    process = pool.Pool(processes=cpu_count() - 1)
    peaks = process.map(func=get_final_size, iterable=range(len(w1)))

    process.close()
    process.join()

    peaks = np.array(peaks)

    # params["peaks"] = peaks[:, 0]
    # params["final_size"] = peaks  # [:, 0]

    # params["peaks"] = pd.Categorical(params.peaks)
    # params["final_size"] = params.final_size/N

    toc = time.time()

    print("Script time is %f" % (toc - tic))

    # %%
    # tmp = get_final_size(w2)

    FS = fsolve(sir_FS, [0.01, 0.5, 1])

    plt.xkcd()
    plt.figure()
    plt.plot(range(len(w1)), peaks, c="b")
    plt.plot([0, len(w1) - 1], [FS[1], FS[1]], ":k")
    plt.xlabel("Idx of parameter values")
    plt.ylabel("Final size")
    plt.show()
