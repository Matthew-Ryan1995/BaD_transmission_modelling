#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 07:09:58 2023

Given the difference in when the peaks occur, is it appropriate to look at the straight difference? I dont
think so.

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

# inf_per = 1
# immune_period = 13
# R0d = 8.4
# todo: Add time direction to plots, fix legend and labels, better color?


def create_params(R0=5, p=1, c=1):
    model_params = load_param_defaults()
    model_params["transmission"] = R0/model_params["infectious_period"]
    model_params["susc_B_efficacy"] = c
    model_params["inf_B_efficacy"] = p

    return model_params


# m_params = create_params(R0=R0d)

# %%

# Sb = 1e-3
# In = 1e-3
# Ib = Rb = 0
# Rn = [0.0, 0.2, 0.4, 0.6]  # , 0.8, 0.95]

# t_start, t_end = [0, 50]

# res_i = list()
# res_s = list()
# M = bad(**m_params)
# for rn in Rn:
#     Sn = 1-Sb-In-Ib-Rb-rn
#     IC = [Sn, Sb, In, Ib, rn, Rb]

#     M.run(IC, t_start, t_end, t_step=0.1)

#     res_i.append(M.get_I())
#     res_s.append(M.get_S())

# %%

# plt.figure()
# plt.title(
#     f"Phase plane, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
# for idx, ii in enumerate(res_i):
#     ss = res_s[idx]
#     plt.plot(ss, ii, label=str(Rn[idx]))
# plt.xlabel('S')
# plt.ylabel('I')
# plt.legend()
# plt.show()

# plt.figure()
# plt.title(
#     f"Time series, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
# for idx, ii in enumerate(res_i):
#     plt.plot(ii, label=str(Rn[idx]))
# plt.xlabel('t')
# plt.ylabel('I')
# plt.legend()
# plt.show()

# %%

def generate_phase_plane(disease_type, save=False):
    if disease_type == "covid_like":
        R0 = 8.4
        title = "Covid-like illness ($\\mathscr{R}_0^D = 8.4$)"
        text_factor = 1.2
    elif disease_type == "flu_like":
        R0 = 1.4
        title = "Flu-like illness ($\\mathscr{R}_0^D = 1.4$)"
        text_factor = 2
    else:
        R0 = disease_type
        title = "$\\mathscr{R}_0^D =$"+f"{R0}"

    m_params = create_params(R0=R0)

    Sb = 1e-3
    In = 1e-3
    Ib = Rb = 0
    # Rn = [0.0, 0.2, 0.4, 0.6]#, 0.8, 0.95]
    Rn = [0.0]  # , 0.5, 0.95]

    # # c_range = np.arange(0 + 1e-5, 1-1e-5, step=1/len(unique_lbls))
    # c_range = np.arange(0 + 1e-5, 1-1e-5, step=1/len(Rn))

    # cmaps = list()

    # for j in c_range:
    #     if j > 0.7:
    #         j += 0.1
    #     cmaps.append(clrs.ListedColormap(plt.cm.rainbow(j)))
    #     # cmaps.append(clrs.ListedColormap(plt.cm.tab20(j)))

    t_start, t_end = [0, 900]

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

        M.run(IC, t_start, t_end, t_step=0.1)
        M2.run(IC, t_start, t_end, t_step=0.1)

        res_i.append(M.get_I())
        res_s.append(M.get_S())
        res_i_n.append(M2.get_I())
        res_s_n.append(M2.get_S())

    plt.figure()
    plt.title(title)
    for idx, ii in enumerate(res_i):
        ss = res_s[idx]
        ii_n = res_i_n[idx]
        ss_n = res_s_n[idx]
        plt.plot(ss, ii, label="Fully protective behaviour", color="black")
        plt.plot(ss_n, ii_n, linestyle=":", color="black",
                 label="No protective behaviour")
    plt.xlabel('S')
    plt.ylabel('I')
    plt.legend()
    plt.show()


generate_phase_plane("covid_like")
generate_phase_plane("flu_like")

# plt.figure()
# plt.title(
#     f"Phase plane diff, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
# for idx, ii in enumerate(res_i):
#     ss = res_s[idx]
#     ii_n = res_i_n[idx]
#     ss_n = res_s_n[idx]
#     plt.plot(ii_n-ii, label=str(Rn[idx]))
#     # plt.plot(ss_n, ii_n, label=str(Rn[idx]) + ", no b")
# plt.xlabel('S')
# plt.ylabel('I')
# plt.legend()
# plt.show()

# plt.figure()
# plt.title(
#     f"Time series comparing, p={m_params['inf_B_efficacy']}, c={m_params['susc_B_efficacy']}")
# for idx, ii in enumerate(res_i):
#     ii_n = res_i_n[idx]
#     plt.plot(ii, label=str(Rn[idx]))
#     plt.plot(ii_n, label=str(Rn[idx]) + ", no b")
# plt.xlabel('t')
# plt.ylabel('I')
# # plt.legend()
# plt.show()

# # %%
# k = -1
# plt.figure()
# plt.plot(res_i[k])
# plt.plot(res_i_n[k])
# plt.show()

# k = 0
# plt.figure()
# plt.plot(res_i[k], res_i_n[k])
# plt.show()
# Sb = 1e-3
# In = 1e-3
# Ib = Rb = 0
# # Rn = [0.0, 0.2, 0.4, 0.6, 0.8]
# Rn = np.arange(0, 1.05, step=0.05)
# Rn[-1] = 1-In-Sb

# t_start, t_end = [0, 600]

# res_diff = list()
# M = bad(**m_params)
# M2 = bad(**m_params)
# M2.update_params(**{"inf_B_efficacy": 0.0, "susc_B_efficacy": 0.0})
# for rn in Rn:
#     Sn = 1-Sb-In-Ib-Rb-rn
#     IC = [Sn, Sb, In, Ib, rn, Rb]

#     M.run(IC, t_start, t_end, incidence=True, t_step=0.1)
#     M2.run(IC, t_start, t_end, incidence=True, t_step=0.1)

#     res_diff.append(M2.incidence[-1] - M.incidence[-1])

# # %%

# plt.figure()
# plt.title(f"Cum diff")
# plt.plot(Rn, np.array(res_diff))
# plt.xlabel("R(0)")
# plt.ylabel("Cum diff")
# plt.show()

# # %%
