#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:45:35 2023

@author: rya200
"""

# %%
import scipy.ndimage
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *
# %%


r0d = np.arange(0.1, 10, step=0.5)
r0b = np.arange(0.1, 5, step=0.5)


def create_params(r0d, r0b, infectious_period=1, immune_period=2):
    params = dict()
    model_params = dict()
    model_params["transmission"] = r0d/infectious_period
    model_params["infectious_period"] = infectious_period
    model_params["immune_period"] = immune_period
    model_params["av_lifespan"] = 0  # Turning off demography
    model_params["susc_B_efficacy"] = 0.7
    model_params["inf_B_efficacy"] = 0.4
    model_params["N_social"] = 0.
    model_params["B_fear"] = 3.5
    model_params["N_const"] = 0.9
    model_params["B_const"] = 0.7
    model_params["B_social"] = r0b / \
        (model_params["N_social"] + model_params["N_const"])

    return model_params


R0b_fixed = 1.5
R0d_fixed = 5.4

ss_results_r0b = list()

for bb in r0b:
    params = create_params(r0d=R0d_fixed, r0b=bb)

    ss, _ = find_ss(params)

    ss_results_r0b.append(ss)

ss_results_r0b = np.array(ss_results_r0b)

ss_results_r0d = list()

for dd in r0d:
    params = create_params(r0d=dd, r0b=R0b_fixed)

    ss, _ = find_ss(params)

    ss_results_r0d.append(ss)

ss_results_r0d = np.array(ss_results_r0d)

# %%

plt.figure()
plt.title("Varying R0b, all ss")
plt.plot(r0b, ss_results_r0b[:, 0], label="Sn")
plt.plot(r0b, ss_results_r0b[:, 1], label="Sb")
plt.plot(r0b, ss_results_r0b[:, 2], label="In")
plt.plot(r0b, ss_results_r0b[:, 3], label="Ib")
plt.plot(r0b, ss_results_r0b[:, 4], label="Rn")
plt.plot(r0b, ss_results_r0b[:, 5], label="Rb")
plt.xlabel("R0b")
plt.ylabel("Steady state")
plt.legend()
plt.show()

# plt.figure()
# plt.title("Varying R0b, I and B only")
# plt.plot(r0b, ss_results_r0b[:, [2, 3]].sum(1), label="I")
# plt.plot(r0b, ss_results_r0b[:, [1, 3, 5]].sum(1), label="B")
# plt.xlabel("R0b")
# plt.ylabel("Steady state")
# plt.legend()
# plt.show()

plt.figure()
plt.title("Varying R0d, all ss")
plt.plot(r0d, ss_results_r0d[:, 0], label="Sn")
plt.plot(r0d, ss_results_r0d[:, 1], label="Sb")
plt.plot(r0d, ss_results_r0d[:, 2], label="In")
plt.plot(r0d, ss_results_r0d[:, 3], label="Ib")
plt.plot(r0d, ss_results_r0d[:, 4], label="Rn")
plt.plot(r0d, ss_results_r0d[:, 5], label="Rb")
plt.xlabel("R0d")
plt.ylabel("Steady state")
plt.legend()
plt.show()

# plt.figure()
# plt.title("Varying R0d, I and B only")
# plt.plot(r0d, ss_results_r0d[:, [2, 3]].sum(1), label="I")
# plt.plot(r0d, ss_results_r0d[:, [1, 3, 5]].sum(1), label="B")
# plt.xlabel("R0d")
# plt.ylabel("Steady state")
# plt.legend()
# plt.show()
