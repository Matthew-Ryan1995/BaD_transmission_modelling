#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 07:09:58 2023

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

inf_per = 7
immune_period = 240
R0d = 5.4

# m_params = dict()
# m_params["transmission"] = R0d/inf_per
# m_params["infectious_period"] = inf_per
# m_params["immune_period"] = immune_period
# m_params["av_lifespan"] = 0  # Turning off demography
# m_params["susc_B_efficacy"] = 1.
# m_params["inf_B_efficacy"] = 1.
# m_params["N_social"] = 0.2
# m_params["B_social"] = .3
# m_params["B_fear"] = 0.7
# m_params["B_const"] = 0.3
# m_params["N_const"] = 0.9


def create_params(R0=5):
    model_params = dict()
    model_params["transmission"] = R0
    model_params["infectious_period"] = 1
    model_params["immune_period"] = 1/0.4
    model_params["av_lifespan"] = 0  # Turning off demography
    model_params["susc_B_efficacy"] = .5
    model_params["inf_B_efficacy"] = .5
    model_params["N_social"] = 1.25
    model_params["B_fear"] = 8.
    model_params["B_const"] = 0.2

    model_params["N_const"] = 0.6
    model_params["B_social"] = 0.4
    # model_params = dict()
    # model_params["transmission"] = R0/inf_per
    # model_params["infectious_period"] = inf_per
    # model_params["immune_period"] = imm_per
    # model_params["av_lifespan"] = 0  # Turning off demography
    # model_params["susc_B_efficacy"] = 1.
    # model_params["inf_B_efficacy"] = 1.
    # model_params["N_social"] = 0.
    # model_params["B_fear"] = 0.
    # model_params["B_const"] = 0.

    # if Bstar == 1:
    #     model_params["N_const"] = 0
    #     model_params["B_social"] = 1
    # else:
    #     model_params["B_social"] = 3.5/(1-Bstar)
    #     model_params["N_const"] = 3.5

    return model_params


m_params = create_params(R0=2.4)

# %%

Sb = 1e-3
In = 1e-3
Ib = Rb = 0
Rn = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]

t_start, t_end = [0, 50]

res_i = list()
res_s = list()
M = bad(**m_params)
for rn in Rn:
    Sn = 1-Sb-In-Ib-Rb-rn
    IC = [Sn, Sb, In, Ib, rn, Rb]

    M.run(IC, t_start, t_end)

    res_i.append(M.get_I())
    res_s.append(M.get_S())

# %%

plt.figure()
plt.title(
    f"Phase plane, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
for idx, ii in enumerate(res_i):
    ss = res_s[idx]
    plt.plot(ss, ii, label=str(Rn[idx]))
plt.xlabel('S')
plt.ylabel('I')
plt.legend()
plt.show()

plt.figure()
plt.title(
    f"Time series, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
for idx, ii in enumerate(res_i):
    plt.plot(ii, label=str(Rn[idx]))
plt.xlabel('t')
plt.ylabel('I')
plt.legend()
plt.show()

# %%

Sb = 1e-3
In = 1e-3
Ib = Rb = 0
Rn = [0.0, 0.95]

t_start, t_end = [0, 50]

res_i = list()
res_s = list()
res_i_n = list()
res_s_n = list()
M = bad(**m_params)
M2 = bad(**m_params)
M2.update_params(**{"inf_B_efficacy": 0.0, "susc_B_efficacy": 0.0})
for rn in Rn:
    Sn = 1-Sb-In-Ib-Rb-rn
    IC = [Sn, Sb, In, Ib, rn, Rb]

    M.run(IC, t_start, t_end)
    M2.run(IC, t_start, t_end)

    res_i.append(M.get_I())
    res_s.append(M.get_S())
    res_i_n.append(M2.get_I())
    res_s_n.append(M2.get_S())

# %%

plt.figure()
plt.title(
    f"Phase plane comparing, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
for idx, ii in enumerate(res_i):
    ss = res_s[idx]
    ii_n = res_i_n[idx]
    ss_n = res_s_n[idx]
    plt.plot(ss, ii, label=str(Rn[idx]))
    plt.plot(ss_n, ii_n, label=str(Rn[idx]) + ", no b")
plt.xlabel('S')
plt.ylabel('I')
plt.legend()
plt.show()

plt.figure()
plt.title(
    f"Time series comparing, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
for idx, ii in enumerate(res_i):
    ii_n = res_i_n[idx]
    plt.plot(ii, label=str(Rn[idx]))
    plt.plot(ii_n, label=str(Rn[idx]) + ", no b")
plt.xlabel('t')
plt.ylabel('I')
plt.legend()
plt.show()

# %%

Sb = 1e-3
In = 1e-3
Ib = Rb = 0
# Rn = [0.0, 0.2, 0.4, 0.6, 0.8]
Rn = np.arange(0, 1.05, step=0.05)
Rn[-1] = 1-In-Sb

t_start, t_end = [0, 600]

res_diff = list()
M = bad(**m_params)
M2 = bad(**m_params)
M2.update_params(**{"inf_B_efficacy": 0.0, "susc_B_efficacy": 0.0})
for rn in Rn:
    Sn = 1-Sb-In-Ib-Rb-rn
    IC = [Sn, Sb, In, Ib, rn, Rb]

    M.run(IC, t_start, t_end, incidence=True)
    M2.run(IC, t_start, t_end, incidence=True)

    res_diff.append(M2.incidence[-1] - M.incidence[-1])

# %%

plt.figure()
plt.title(f"Cum diff")
plt.plot(Rn, np.array(res_diff))
plt.xlabel("R(0)")
plt.ylabel("Cum diff")
plt.show()

# %%
