#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:42:34 2023

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

bifurc_params = dict()
bifurc_params["transmission"] = 3/2
bifurc_params["infectious_period"] = 1/1
bifurc_params["immune_period"] = 1/0.5
bifurc_params["av_lifespan"] = 0  # Turning off demography
bifurc_params["susc_B_efficacy"] = 0.5
bifurc_params["inf_B_efficacy"] = 0.3
bifurc_params["N_social"] = 0.2
bifurc_params["N_fear"] = 1.1
bifurc_params["B_social"] = 1.3
bifurc_params["B_fear"] = 0.5
bifurc_params["B_const"] = 0.7
bifurc_params["N_const"] = 0.9

var = "inf_B_efficacy"

r0_b = np.arange(0, 5.1, step=0.1)
plt.figure()
for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:

    I = list()
    B = list()
    tmp = dict(bifurc_params)
    tmp[var] = p
    for idx in range(len(r0_b)):
        ww = r0_b[idx] * (tmp["N_social"] + tmp["N_fear"] + tmp["N_const"])
        tmp["B_social"] = ww

        ss, _ = find_ss(tmp)
        I.append(ss[[2, 3]].sum())
        B.append(ss[[1, 3, 5]].sum())

    II = np.array(I)
    BB = np.array(B)

    plt.title("I")
    plt.plot(r0_b, II, label=f"{p}")
plt.legend()
plt.show()

w2_range = np.arange(0, 5.1, step=0.1)
plt.figure()
for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:

    I = list()
    B = list()
    tmp = dict(bifurc_params)
    tmp[var] = p
    for idx in range(len(w2_range)):
        ww = w2_range[idx]
        tmp["B_fear"] = ww

        ss, _ = find_ss(tmp)
        I.append(ss[[2, 3]].sum())
        B.append(ss[[1, 3, 5]].sum())

    II = np.array(I)
    BB = np.array(B)

    plt.title("I: w2 vary")
    plt.plot(r0_b, II, label=f"{p}")
plt.ylabel("I")
plt.legend()
plt.show()

w3_range = np.arange(0, 5.1, step=0.1)
plt.figure()
for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:

    I = list()
    B = list()
    tmp = dict(bifurc_params)
    tmp[var] = p
    for idx in range(len(w3_range)):
        ww = w3_range[idx]
        tmp["B_const"] = ww

        ss, _ = find_ss(tmp)
        I.append(ss[[2, 3]].sum())
        B.append(ss[[1, 3, 5]].sum())

    II = np.array(I)
    BB = np.array(B)

    plt.title("I: w3 vary")
    plt.plot(r0_b, II, label=f"{p}")
plt.ylabel("I")
plt.legend()
plt.show()

r0_b = np.arange(0, 5.1, step=0.1)
plt.figure()
for ff in range(1):
    I = list()
    B = list()
    tmp = dict(bifurc_params)
    if ff == 0:
        tmp["susc_B_efficacy"] = 0
        tmp["inf_B_efficacy"] = 0
        lbl = "No behavioural affect"
    if ff == 1:
        tmp["susc_B_efficacy"] = 0.5
        tmp["inf_B_efficacy"] = 0
        lbl = "S only"
    if ff == 2:
        tmp["susc_B_efficacy"] = 0.
        tmp["inf_B_efficacy"] = 0.5
        lbl = "I only"
    if ff == 3:
        tmp["susc_B_efficacy"] = 0.5
        tmp["inf_B_efficacy"] = 0.5
        lbl = "Both"
    for idx in range(len(r0_b)):
        ww = r0_b[idx] * (tmp["N_social"] + tmp["N_fear"] + tmp["N_const"])
        tmp["B_social"] = ww

        ss, _ = find_ss(tmp)
        I.append(ss[[2, 3]].sum())
        B.append(ss[[1, 3, 5]].sum())

    II = np.array(I)
    BB = np.array(B)

    plt.title("I")
    plt.plot(r0_b, II, label=lbl)
plt.legend()
plt.show()

# plt.figure()
# plt.title("B")
# plt.plot(r0_b, BB, "b")
# plt.show()

# plt.figure()
# plt.title("I vs B")
# plt.plot(BB, II, "g")
# plt.show()


# Different set of params
# bifurc_params = dict()
# bifurc_params["transmission"] = 1
# bifurc_params["infectious_period"] = 1/0.4
# bifurc_params["immune_period"] = 1/(8*30)
# bifurc_params["av_lifespan"] = 0  # Turning off demography
# bifurc_params["susc_B_efficacy"] = 0.4
# bifurc_params["inf_B_efficacy"] = 0.8
# bifurc_params["N_social"] = 0.5
# bifurc_params["N_fear"] = 0.0
# bifurc_params["B_social"] = 0.05 * 8
# bifurc_params["B_fear"] = 8
# bifurc_params["B_const"] = 0.01
# bifurc_params["N_const"] = 0.01

# M2 = bad(**bifurc_params)

# multi_val = M2.Rzero()/M2.transmission

# # multi_val = 1/bifurc_params["infectious_period"]

# beta_vals = np.arange(1., 8.1, step=0.1)


# plt.figure()
# for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
#     res = list()
#     bifurc_params["inf_B_efficacy"] = p
#     M2 = bad(**bifurc_params)

#     multi_val = M2.Rzero()/M2.transmission
#     for idxx in range(len(beta_vals)):
#         b = beta_vals[idxx] / multi_val
#         bifurc_params["transmission"] = b
#         ss, _ = find_ss(bifurc_params)

#         # ss[ss.round(4) > 0] = 1
#         res.append(ss.round(4))

#     res = np.array(res)

#     # plt.ylim([0, 1])
#     plt.title("B SS")
#     # plt.plot(beta_vals, res[:, 1] + res[:, 3] + res[:, 5], label=f"{p}")
#     plt.plot(beta_vals, res[:, 2] + res[:, 3], label=f"{p}")
# plt.legend()
# plt.show()

# plt.figure()
# plt.ylim([0, 1])
# plt.title("All SS")
# plt.plot(beta_vals, res[:, 0], "green", label="Sn")
# plt.plot(beta_vals, res[:, 1], "black", label="Sb")
# plt.plot(beta_vals, res[:, 2], "red", label="In")
# plt.plot(beta_vals, res[:, 3], "magenta", label="Ib")
# plt.plot(beta_vals, res[:, 4], "blue", label="Rn")
# plt.plot(beta_vals, res[:, 5], "cyan", label="Rb")
# plt.legend()
# plt.show()

# plt.figure()
# plt.ylim([0, 1])
# plt.title("I SS")
# plt.plot(beta_vals, res[:, 2] + res[:, 3], "red", label="I")
# plt.legend()
# plt.show()

# plt.figure()
# plt.ylim([0, 1])
# plt.title("B SS")
# plt.plot(beta_vals, res[:, 1] + res[:, 3] + res[:, 5], "black", label="Sb")
# plt.legend()
# plt.show()
