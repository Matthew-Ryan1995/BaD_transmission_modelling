#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:00:19 2023

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

P = 1
Ib0, Rb0, Rn0 = np.zeros(3)
Sb0 = 1e-3  # 1 in a million seeded with behaviour
In0 = 1e-3  # 1 in a million seeded with disease
# Ib0, Rb0, Rn0 = np.zeros(3)
# Sb0 = 1-0.6951793156273507  # 1 in a million seeded with behaviour
# In0 = Ib0 = 1e-6  # 1 in a million seeded with disease

Sn0 = P - Sb0 - Ib0 - Rb0 - In0 - Rn0

PP = np.array([Sn0, Sb0, In0, Ib0, Rn0, Rb0])

w1 = 8
R0_d = 3
R0_b = 1.01
gamma = 1/7

cust_params = dict()
cust_params["transmission"] = R0_d*gamma
cust_params["infectious_period"] = 1/gamma
cust_params["immune_period"] = 240
cust_params["av_lifespan"] = 0  # Turning off demography
cust_params["susc_B_efficacy"] = 0.9
cust_params["inf_B_efficacy"] = 0.9
cust_params["N_social"] = 0.5
cust_params["N_const"] = 0.01
cust_params["B_social"] = R0_b * \
    (cust_params["N_social"] + cust_params["N_const"])
cust_params["B_fear"] = 0.01  # w1
cust_params["B_const"] = 0.0

# cust_params["transmission"] = R0*gamma
# cust_params["infectious_period"] = 1/gamma
# cust_params["immune_period"] = 240
# cust_params["av_lifespan"] = 0  # Turning off demography
# cust_params["susc_B_efficacy"] = 0.8
# cust_params["inf_B_efficacy"] = 0.4
# cust_params["N_social"] = 0.
# cust_params["B_social"] = 0.0
# cust_params["B_fear"] = 0
# cust_params["B_const"] = 0.0
# cust_params["N_const"] = 0.0

M1 = bad(**cust_params)

# M1.run(IC=PP, t_start=0, t_end=900, t_step=1)

w2 = [0.0, .05, .5, 1.0]

fig_i, ax_i = plt.subplots()
plt.title("Disease prevalence")
plt.xlabel("Time")
plt.ylabel("Disease prevalence")

fig_b, ax_b = plt.subplots()
plt.title("Behaviour prevalence")
plt.xlabel("Time")
plt.ylabel("Behaviour prevalence")

fig_phase, ax_phase = plt.subplots()
plt.title("BaD phase plane")
plt.xlabel("Disease prevalence")
plt.ylabel("Behaviour prevalence")

fig_phase2, ax_phase2 = plt.subplots()
plt.title("S/I phase plane")
plt.ylabel("Disease prevalence")
plt.xlabel("S prevalence")


for ww in w2:
    M1.update_params(**{"B_fear": ww})
    # if ww == 0:
    #     M1.update_params(**{"B_social": ww})
    # else:
    #     M1.update_params(**{"B_social": cust_params["B_social"]})
    M1.run(IC=PP, t_start=0, t_end=1900, t_step=1)
    tt = M1.t_range[0:(len(M1.results))]

    ax_i.plot(tt, M1.get_I(), label=f"$\\omega_2 =${ww}")
    ax_b.plot(tt, M1.get_B(), label=f"$\\omega_2 =${ww}")
    ax_phase.plot(M1.get_I(), M1.get_B(), label=f"$\\omega_2 =${ww}")
    ax_phase2.plot(M1.results[:, [
                   0, 1]].sum(1), M1.get_I(), label=f"$\\omega_2 =${ww}")

fig_i.legend()
fig_b.legend()
fig_phase.legend()
fig_phase2.legend()

# fig_i.show()
# fig_b.show()
# fig_phase.show()
