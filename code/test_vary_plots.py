#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 08:46:42 2023

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


def ss_vary_r0(params, R0_range=[0, 10], R0_step=0.1):
    # params = dict()
    # params["transmission"] = 3/2
    # params["infectious_period"] = 1/1
    # params["immune_period"] = 1/0.5
    # params["av_lifespan"] = 0  # Turning off demography
    # params["susc_B_efficacy"] = 0.5
    # params["inf_B_efficacy"] = 0.3
    # params["N_social"] = 0.2
    # params["N_fear"] = 1.1
    # params["B_social"] = 1.3
    # params["B_fear"] = 0.5
    # params["B_const"] = 0.7
    # params["N_const"] = 0.9

    R0 = np.arange(R0_range[0], R0_range[1] + R0_step, step=R0_step)

    M_base = bad(**params)

    # trans_val = M_base.Rzero()/M_base.transmission
    # trans_val = 1/trans_val
    trans_val = 1/M_base.infectious_period

    res = list()

    modify_dict = dict(params)

    for r0 in R0:
        modify_dict["transmission"] = r0*trans_val
        ss, _ = find_ss(modify_dict)
        res.append(ss)

    res = np.array(res)
    return res, R0


def r0_v_r0D(params, R0_range=[0, 5], R0_step=0.1):
    R0 = np.arange(R0_range[0], R0_range[1] + R0_step, step=R0_step)

    M_base = bad(**params)

    # trans_val = M_base.Rzero()/M_base.transmission
    # trans_val = 1/trans_val
    trans_val = 1/M_base.infectious_period

    res = list()

    modify_dict = dict(params)

    for r0 in R0:
        modify_dict["transmission"] = r0*trans_val
        M_base.update_params(**modify_dict)
        res.append(M_base.Rzero())

    res = np.array(res)
    return res, R0


def r0_v_r0B(params, R0_range=[0, 5], R0_step=0.1):
    R0 = np.arange(R0_range[0], R0_range[1] + R0_step, step=R0_step)

    M_base = bad(**params)

    # trans_val = M_base.Rzero()/M_base.transmission
    # trans_val = 1/trans_val
    trans_val = M_base.N_social + M_base.N_fear + \
        M_base.N_const  # 1/M_base.infectious_period

    res = list()

    modify_dict = dict(params)

    for r0 in R0:
        modify_dict["B_social"] = r0*trans_val
        M_base.update_params(**modify_dict)
        res.append(M_base.Rzero())

    res = np.array(res)
    return res, R0

# %%


bifurc_params = dict()
bifurc_params["transmission"] = 3/2
bifurc_params["infectious_period"] = 1/1
bifurc_params["immune_period"] = 1/0.5
bifurc_params["av_lifespan"] = 0  # Turning off demography
bifurc_params["susc_B_efficacy"] = 0.5
bifurc_params["inf_B_efficacy"] = 0.3
bifurc_params["N_social"] = 1  # 0.2
bifurc_params["N_fear"] = 0.1  # 1.1
bifurc_params["B_social"] = 0  # 1.3
bifurc_params["B_fear"] = 0  # 0.5
bifurc_params["B_const"] = 0  # 0.7
bifurc_params["N_const"] = 0.5  # 0.9


# %%
behave_params = {"B_social": (2, "$\\omega_1$"),
                 "B_fear": (2, "$\\omega_2$"),
                 "B_const": (2, "$\\omega_3$")}

lbl_to_symb = {
    "infectious_period": "$1/\\gamma$",
    "immune_period": "$1/\\nu$",
    "susc_B_efficacy": "c",
    "inf_B_efficacy": "p",
    "N_social": "$\\alpha_1$",
    "N_fear": "$\\alpha_2$",
    "N_const": "$\\alpha_3$"
}

caption = "| "
for lbl in lbl_to_symb.keys():
    if lbl == "inf_B_efficacy":
        caption += "\n| "
    caption += lbl_to_symb[lbl] + "=" + str(bifurc_params[lbl]) + " | "

plt.figure()
plt.title(
    "Infection rates against disease characteristic \nfor different behaviour parameters")
plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
plt.ylabel("I")

# No behaviour
vary_params = dict(bifurc_params)
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [2, 3]].sum(1), label="No behaviour")

for f1 in behave_params.keys():
    vary_params = dict(bifurc_params)
    vary_params[f1] = behave_params[f1][0]
    res, _ = ss_vary_r0(vary_params)

    plt.plot(R0, res[:, [2, 3]].sum(1),
             label=f"{behave_params[f1][1]}= {behave_params[f1][0]}")

plt.text(-0.4, -0.05, caption,  va="top")
plt.legend(loc=[1.05, 0.25])
plt.savefig("../img/I_by_behaviour/I_switch_on_omega.png",
            dpi=600, bbox_inches="tight")
plt.show()

plt.figure()
plt.title(
    "Behaviour rates against disease characteristic \nfor different behaviour parameters")
plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
plt.ylabel("B")

# No behaviour
vary_params = dict(bifurc_params)
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [1, 3, 5]].sum(1), label="No behaviour")

for f1 in behave_params.keys():
    vary_params = dict(bifurc_params)
    vary_params[f1] = behave_params[f1][0]
    res, _ = ss_vary_r0(vary_params)

    plt.plot(R0, res[:, [1, 3, 5]].sum(1),
             label=f"{behave_params[f1][1]}= {behave_params[f1][0]}")

plt.text(-0.4, -0.1, caption,  va="top")
plt.legend(loc=[1.05, 0.25])
plt.savefig("../img/I_by_behaviour/B_switch_on_omega.png",
            dpi=600, bbox_inches="tight")
plt.show()


plt.figure()
plt.title("Infection rates against disease characteristic \nfor different behaviour parameters")
plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
plt.ylabel("I")

# No behaviour
vary_params = dict(bifurc_params)
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [2, 3]].sum(1), label="No behaviour")

var1 = "B_social"

vary_params[var1] = behave_params[var1][0]
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [2, 3]].sum(1),
         label=f"{behave_params[var1][1]}= {behave_params[var1][0]}")


for f1 in behave_params.keys():
    if f1 == var1:
        continue
    tmp_params = dict(vary_params)
    tmp_params[f1] = behave_params[f1][0]
    res, _ = ss_vary_r0(tmp_params)

    plt.plot(R0, res[:, [2, 3]].sum(1),
             label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params[f1][1]}= {behave_params[f1][0]}")

vary_params["B_fear"] = behave_params["B_fear"][0]
vary_params["B_const"] = behave_params["B_const"][0]
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [2, 3]].sum(1),
         label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")
vary_params["B_social"] = 0
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [2, 3]].sum(1),
         label=f"{behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")

plt.text(-0.4, -0.05, caption,  va="top")
plt.legend(loc=[1.05, 0.25])
plt.savefig("../img/I_by_behaviour/I_sadd_to_omega_1.png",
            dpi=600, bbox_inches="tight")
plt.show()

plt.figure()
plt.title("Behaviour rates against disease characteristic \nfor different behaviour parameters")
plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
plt.ylabel("B")

# No behaviour
vary_params = dict(bifurc_params)
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [1, 3, 5]].sum(1), label="No behaviour")

var1 = "B_social"

vary_params[var1] = behave_params[var1][0]
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [1, 3, 5]].sum(1),
         label=f"{behave_params[var1][1]}= {behave_params[var1][0]}")


for f1 in behave_params.keys():
    if f1 == var1:
        continue
    tmp_params = dict(vary_params)
    tmp_params[f1] = behave_params[f1][0]
    res, _ = ss_vary_r0(tmp_params)

    plt.plot(R0, res[:, [1, 3, 5]].sum(1),
             label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params[f1][1]}= {behave_params[f1][0]}")

vary_params["B_fear"] = behave_params["B_fear"][0]
vary_params["B_const"] = behave_params["B_const"][0]
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [1, 3, 5]].sum(1),
         label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")
vary_params["B_social"] = 0
res, R0 = ss_vary_r0(vary_params)
plt.plot(R0, res[:, [1, 3, 5]].sum(1),
         label=f"{behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")

plt.text(-0.4, -0.1, caption,  va="top")
plt.legend(loc=[1.05, 0.25])
plt.savefig("../img/I_by_behaviour/B_sadd_to_omega_1.png",
            dpi=600, bbox_inches="tight")
plt.show()


# # Social only
# vary_params = dict(bifurc_params)
# vary_params["B_social"] = 2
# res, _ = ss_vary_r0(vary_params)

# plt.plot(R0, res[:, [2, 3]].sum(1), label=f"$\\omega_1$ = {2}")

# # Fear only
# vary_params = dict(bifurc_params)
# vary_params["B_fear"] = 1
# res, _ = ss_vary_r0(vary_params)

# plt.plot(R0, res[:, [2, 3]].sum(1), label=f"$\\omega_2$ = {1}")

# # Const only
# vary_params = dict(bifurc_params)
# vary_params["B_const"] = 0.5
# res, _ = ss_vary_r0(vary_params)

# plt.plot(R0, res[:, [2, 3]].sum(1), label=f"$\\omega_3$ = {0.5}")


# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("R0 vs R0B for different beta")
# # for w1 in np.arange(1, 6, step=1).round(2):
# #     vary_params["transmission"] = w1
# #     res, R0 = r0_v_r0B(vary_params)
# #     plt.plot(R0, res, label=f"beta = {w1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0^{B}$")
# # plt.ylabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()
# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("R0 vs R0D for different w1")
# # for w1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_social"] = w1
# #     res, R0 = r0_v_r0D(vary_params)
# #     plt.plot(R0, res, label=f"w1 = {w1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0^{D}$")
# # plt.ylabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # plt.figure()
# # plt.title("Full bifurication diagram")
# # plt.plot(R0, res[:, 0], label="Sn")
# # plt.plot(R0, res[:, 1], label="Sb")
# # plt.plot(R0, res[:, 2], label="In")
# # plt.plot(R0, res[:, 3], label="Ib")
# # plt.plot(R0, res[:, 4], label="Rn")
# # plt.plot(R0, res[:, 5], label="Rb")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Infection bifurication diagram")
# # for w1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_social"] = w1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, 2] + res[:, 3], label=f"w1 = {w1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Behaviour bifurication diagram")
# # for w1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_social"] = w1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, [1, 3, 5]].sum(1),  label=f"w1 = {w1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Infection bifurication diagram")
# # for w2 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_fear"] = w2
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, 2] + res[:, 3], label=f"w2 = {w2}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Behaviour bifurication diagram")
# # for w2 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_fear"] = w2
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, [1, 3, 5]].sum(1),  label=f"w2 = {w2}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Infection bifurication diagram")
# # for w3 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_const"] = w3
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, 2] + res[:, 3], label=f"w3 = {w3}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Behaviour bifurication diagram")
# # for w3 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["B_const"] = w3
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, [1, 3, 5]].sum(1),  label=f"w3 = {w3}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Infection bifurication diagram")
# # for a1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["N_social"] = a1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, 2] + res[:, 3], label=f"a1 = {a1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Behaviour bifurication diagram")
# # for a1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["N_social"] = a1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, [1, 3, 5]].sum(1),  label=f"a1 = {a1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Infection bifurication diagram")
# # for a1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["N_fear"] = a1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, 2] + res[:, 3], label=f"a2 = {a1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Behaviour bifurication diagram")
# # for a1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["N_fear"] = a1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, [1, 3, 5]].sum(1),  label=f"a2 = {a1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Infection bifurication diagram")
# # for a1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["N_const"] = a1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, 2] + res[:, 3], label=f"a3 = {a1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # vary_params = dict(bifurc_params)
# # plt.figure()
# # plt.title("Behaviour bifurication diagram")
# # for a1 in np.arange(0, 1.1, step=0.25).round(2):
# #     vary_params["N_const"] = a1
# #     res, R0 = ss_vary_r0(vary_params)
# #     plt.plot(R0, res[:, [1, 3, 5]].sum(1),  label=f"a3 = {a1}")
# # plt.legend()
# # plt.xlabel("$\\mathscr{R}_0$")
# # # plt.ylim([0, 1])
# # plt.show()

# # # plt.figure()
# # # plt.title("Behaviour-Infection phase plane with increased $\\mathscr{R}_0$")
# # # plt.plot(res[:, [2, 3]].sum(1), res[:, [1, 3, 5]].sum(1))
# # # plt.xlabel("Infection")
# # # plt.ylabel("Behaviour")
# # # plt.show()
