#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 15:00:54 2023

@author: rya200
"""

from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *
import json


# %%

# %%

R0 = 8.4
gamma = 1

nu = 35

model_params = load_param_defaults()
model_params["transmission"] = R0 / gamma
model_params["infectious_period"] = gamma
model_params["immune_period"] = nu
model_params["susc_B_efficacy"] = 1  # .5
model_params["inf_B_efficacy"] = 1  # .5

# model_params["N_social"] = 1.25
# model_params["B_fear"] = 8.
# model_params["B_const"] = 0.2
# model_params["N_const"] = 0.6
# model_params["B_social"] = 0.4


def sir_odes(t, PP):
    Y = np.zeros(3)

    Y[0] = -model_params["transmission"] * PP[0] * \
        PP[1] + (1/model_params["immune_period"]) * PP[2]
    Y[1] = model_params["transmission"] * PP[0] * PP[1] - \
        (1/model_params["infectious_period"]) * PP[1]
    Y[2] = (1/model_params["infectious_period"]) * PP[1] - \
        (1/model_params["immune_period"]) * PP[2]

    return Y

# %%


IC = [1-1e-3, 1e-3, 0]
t_span = [0, 600]
t_eval = np.arange(t_span[0], t_span[1])

# %%
sir_results = solve_ivp(fun=sir_odes, t_span=t_span, y0=IC, t_eval=t_eval)


PP = sir_results.y.T[-1, :]
PP0 = [PP[0] + 1e-3, 0, PP[1], 0, PP[2] - 1e-3, 0]
# %%
M = bad(**model_params)
M.run(IC=PP0, t_start=t_span[0], t_end=t_span[1])

plt.figure()
plt.title("Initial SIR")
plt.plot(sir_results.y.T[:, 0], label="S")
plt.plot(sir_results.y.T[:, 1], label="I")
plt.plot(sir_results.y.T[:, 2], label="R")
plt.legend()
plt.show()

plt.figure()
plt.title("BaD starting at SIR")
plt.plot(M.get_S(), label="S")
plt.plot(M.get_I(), label="I")
plt.plot(1-M.get_S() - M.get_I(), label="R")
plt.legend()
plt.show()
