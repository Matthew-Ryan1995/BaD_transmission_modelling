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


def ss_vary_r0(params, R0_range=[0.01, 10], R0_step=0.1):
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
bifurc_params["B_social"] = 0.5  # 1.3
bifurc_params["B_fear"] = 0.5  # 0.5
bifurc_params["B_const"] = 0.5  # 0.7
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

# plt.figure()
# plt.title(
#     "Infection rates against disease characteristic \nfor different behaviour parameters")
# plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
# plt.ylabel("I")

# # No behaviour
# vary_params = dict(bifurc_params)
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [2, 3]].sum(1), label="No behaviour")

# w1_values = np.arange(0, 3, step=0.1)

# for w1 in w1_values:
#     vary_params = dict(bifurc_params)
#     vary_params["B_const"] = w1
#     res, _ = ss_vary_r0(vary_params)

#     plt.plot(R0, res[:, [2, 3]].sum(1),
#              label=f"w1 = {w1}")

# plt.text(-0.4, -0.05, caption,  va="top")
# # plt.legend(loc=[1.05, 0.25])
# # plt.savefig("../img/I_by_behaviour/I_switch_on_omega.png",
# # dpi = 600, bbox_inches = "tight")
# plt.show()

vary_params = dict(bifurc_params)
vary_params["B_const"] = 0
res, R0 = ss_vary_r0(vary_params)

v1 = res[:, [2, 3]].sum(1)
vary_params = dict(bifurc_params)
vary_params["B_const"] = 3
res, _ = ss_vary_r0(vary_params)

v2 = res[:, [2, 3]].sum(1)

vary_params = dict(bifurc_params)
vary_params["B_social"] = 3
res, _ = ss_vary_r0(vary_params)

v3 = res[:, [2, 3]].sum(1)

vary_params = dict(bifurc_params)
vary_params["B_fear"] = 3
res, _ = ss_vary_r0(vary_params)

v4 = res[:, [2, 3]].sum(1)

# %%
plt.figure()
plt.plot(R0, v1, "black")
plt.plot(R0, v2, "green", label="w3")
plt.plot(R0, v3, "red", label="w1")
plt.plot(R0, v4, "blue", label="w2")
plt.fill_between(R0, v1, v2, alpha=0.25, color="green")
plt.fill_between(R0, v1, v3, alpha=0.25, color="red")
plt.fill_between(R0, v1, v4, alpha=0.25, color="blue")
plt.legend()
plt.show()

# %%

bifurc_params2 = dict()
bifurc_params2["transmission"] = 3/2
bifurc_params2["infectious_period"] = 1/1
bifurc_params2["immune_period"] = 1/0.5
bifurc_params2["av_lifespan"] = 0  # Turning off demography
bifurc_params2["susc_B_efficacy"] = 0.
bifurc_params2["inf_B_efficacy"] = 0.
bifurc_params2["N_social"] = 0.2
bifurc_params2["N_fear"] = 1.1
bifurc_params2["B_social"] = 1.
bifurc_params2["B_fear"] = 0.5
bifurc_params2["B_const"] = 0.1
bifurc_params2["N_const"] = 0.9


vary_params = dict(bifurc_params2)
vary_params["susc_B_efficacy"] = 0
res, R0 = ss_vary_r0(vary_params)

v1 = res[:, [2, 3]].sum(1)
vary_params = dict(bifurc_params2)
vary_params["susc_B_efficacy"] = 1
res, _ = ss_vary_r0(vary_params)

v2 = res[:, [2, 3]].sum(1)

vary_params = dict(bifurc_params2)
vary_params["inf_B_efficacy"] = 1
res, _ = ss_vary_r0(vary_params)

v3 = res[:, [2, 3]].sum(1)

# %%
plt.figure()
plt.plot(R0, v1, "black")
plt.plot(R0, v2, "green")
plt.fill_between(R0, v1, v2, alpha=0.25, color="green")
plt.show()

plt.figure()
plt.plot(R0, v1, "black")
plt.plot(R0, v3, "red")

plt.fill_between(R0, v1, v3, alpha=0.25, color="red")
plt.show()

# %%


def return_params(Istar):
    behave_rate = 1/(1-Istar)

    days_behave = 7

    new_params = dict()
    new_params["transmission"] = 1/5
    new_params["infectious_period"] = 5
    new_params["immune_period"] = 8*30
    new_params["av_lifespan"] = 0  # Turning off demography
    new_params["susc_B_efficacy"] = 0.
    new_params["inf_B_efficacy"] = 0.
    new_params["N_social"] = 0.
    new_params["N_fear"] = 0
    new_params["B_social"] = behave_rate/days_behave
    new_params["B_fear"] = 0.
    new_params["B_const"] = 0.
    new_params["N_const"] = 1/days_behave

    return new_params


params1 = return_params(0.1)
params2 = return_params(0.5)
params3 = return_params(0.9)


# M_01_noprotect = bad(**params1)

res, R0 = ss_vary_r0(params1)

I_01_noproc = res[:, [2, 3]].sum(1)

params1["susc_B_efficacy"] = 1
params1["inf_B_efficacy"] = 1
# M_01_protect = bad(**params1)

res, R0 = ss_vary_r0(params1)

I_01_proc = res[:, [2, 3]].sum(1)


# M_05_noprotect = bad(**params2)

res, R0 = ss_vary_r0(params2)


I_05_noproc = res[:, [2, 3]].sum(1)

params2["susc_B_efficacy"] = 1
params2["inf_B_efficacy"] = 1
M_05_protect = bad(**params2)

res, R0 = ss_vary_r0(params2)

I_05_proc = res[:, [2, 3]].sum(1)


# M_09_noprotect = bad(**params3)
res, R0 = ss_vary_r0(params3)

I_09_noproc = res[:, [2, 3]].sum(1)

params3["susc_B_efficacy"] = 1
params3["inf_B_efficacy"] = 1
M_09_potect = bad(**params3)

res, R0 = ss_vary_r0(params3)

I_09_proc = res[:, [2, 3]].sum(1)

# %%

plt.figure()
plt.title("B = 0.1")
plt.plot(R0, I_01_noproc, "black", label="No protection from behaviour")
plt.plot(R0, I_01_proc, "red", label="Full from behaviour")
plt.fill_between(R0, I_01_noproc, I_01_proc, color="red", alpha=0.1)
plt.xlabel("beta/gamma")
plt.ylabel("Istar")
plt.show()

plt.figure()
plt.title("B = 0.5")
plt.plot(R0, I_05_noproc, "black", label="No protection from behaviour")
plt.plot(R0, I_05_proc, "blue", label="Full from behaviour")
plt.fill_between(R0, I_05_noproc, I_05_proc, color="blue", alpha=0.1)
plt.xlabel("beta/gamma")
plt.ylabel("Istar")
plt.show()

plt.figure()
plt.title("B = 0.9")
plt.plot(R0, I_09_noproc, "black", label="No protection from behaviour")
plt.plot(R0, I_09_proc, "green", label="Full from behaviour")
plt.fill_between(R0, I_09_noproc, I_09_proc, color="green", alpha=0.1)
plt.xlabel("beta/gamma")
plt.ylabel("Istar")
plt.show()

# %%


new_params = dict()
new_params["transmission"] = 5/5
new_params["infectious_period"] = 5
new_params["immune_period"] = 8*30
new_params["av_lifespan"] = 0  # Turning off demography
new_params["susc_B_efficacy"] = 0.
new_params["inf_B_efficacy"] = 0.
new_params["N_social"] = 1.3
new_params["N_fear"] = 1.
new_params["B_social"] = 1
new_params["B_fear"] = .5
new_params["B_const"] = 0.01
new_params["N_const"] = 0.01

new_params_full_proc = dict(new_params)
new_params_full_proc["susc_B_efficacy"] = 1
new_params_full_proc["inf_B_efficacy"] = 1

R0d = np.arange(1.1, 10.1, .1)
R0b = np.arange(0.1, 2.1, .1)

mesh = np.meshgrid(R0d, R0b)

pars = np.array(mesh).reshape(2, len(R0d) * len(R0b)).T

diff_list = list()
IC = [1-1e-3, 0, 1e-3, 0, 0, 0]
t_start, t_end = [0, 100]

tot_behave = list()


for idx in range(pars.shape[0]):
    new_params["transmission"] = pars[idx, 0] / new_params["infectious_period"]
    new_params["B_social"] = pars[idx, 1] * \
        (new_params["N_social"] + new_params["N_const"] + new_params["N_fear"])

    new_params_full_proc = dict(new_params)
    new_params_full_proc["susc_B_efficacy"] = 1
    new_params_full_proc["inf_B_efficacy"] = 1

    ss, _ = find_ss(new_params_full_proc)

    tot_behave.append(ss[[1, 3, 5]].sum())

    M1 = bad(**new_params)
    M2 = bad(**new_params_full_proc)

    M1.run(IC, t_start, t_end)
    M2.run(IC, t_start, t_end)

    diff_list.append((M1.get_I().max() - M2.get_I().max())/(M1.get_I().max()))

diff = np.array(diff_list).reshape(mesh[0].shape)
tot_behave = np.array(tot_behave).reshape(mesh[0].shape)
# %%
R0_1_list = list()
for rr in R0b:
    new_params["B_social"] = rr * \
        (new_params["N_social"] + new_params["N_const"] + new_params["N_fear"])

    new_params_full_proc = dict(new_params)
    new_params_full_proc["transmission"] = 1/new_params["infectious_period"]
    new_params_full_proc["susc_B_efficacy"] = 1
    new_params_full_proc["inf_B_efficacy"] = 1
    M2 = bad(**new_params_full_proc)
    tmp = M2.Rzero()
    R0_1_list.append(1/tmp)

R0_1 = np.array(R0_1_list)
# %%
plt.figure()
plt.title("Relative difference in largest peak of infection")
plt.imshow(diff,
           origin='lower',
           extent=[mesh[0].min(), mesh[0].max(), mesh[1].min(), mesh[1].max()],
           aspect="auto", vmin=0, vmax=1)
plt.plot(R0_1, R0b, "black")
plt.xlim(1.1, 10)
plt.xlabel("Disease characteristic")
plt.xlabel("Behaviour characteristic")
plt.colorbar()
plt.show()

plt.figure()
plt.title("Corresponding behaviour under full protection")
plt.imshow(tot_behave,
           origin='lower',
           extent=[mesh[0].min(), mesh[0].max(), mesh[1].min(), mesh[1].max()],
           aspect="auto")
plt.xlabel("Disease characteristic")
plt.xlabel("Behaviour characteristic")
plt.colorbar()
plt.show()

# M1 = bad(**new_params)

# IC = [1-1e-3, 0, 1e-3, 0, 0, 0]
# t_start, t_end = [0, 300]

# M1.run(IC, t_start, t_end)

# plt.figure()
# plt.plot(M1.results[:, 2], "r")
# plt.plot(M1.results[:, 3], "r:")
# plt.show()

# plt.figure()
# plt.title(
#     "Behaviour rates against disease characteristic \nfor different behaviour parameters")
# plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
# plt.ylabel("B")

# # No behaviour
# vary_params = dict(bifurc_params)
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [1, 3, 5]].sum(1), label="No behaviour")

# for f1 in behave_params.keys():
#     vary_params = dict(bifurc_params)
#     vary_params[f1] = behave_params[f1][0]
#     res, _ = ss_vary_r0(vary_params)

#     plt.plot(R0, res[:, [1, 3, 5]].sum(1),
#              label=f"{behave_params[f1][1]}= {behave_params[f1][0]}")

# plt.text(-0.4, -0.1, caption,  va="top")
# plt.legend(loc=[1.05, 0.25])
# # plt.savefig("../img/I_by_behaviour/B_switch_on_omega.png",
# # dpi=600, bbox_inches="tight")
# plt.show()


# plt.figure()
# plt.title("Infection rates against disease characteristic \nfor different behaviour parameters")
# plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
# plt.ylabel("I")

# # No behaviour
# vary_params = dict(bifurc_params)
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [2, 3]].sum(1), label="No behaviour")

# var1 = "B_social"

# vary_params[var1] = behave_params[var1][0]
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [2, 3]].sum(1),
#          label=f"{behave_params[var1][1]}= {behave_params[var1][0]}")


# for f1 in behave_params.keys():
#     if f1 == var1:
#         continue
#     tmp_params = dict(vary_params)
#     tmp_params[f1] = behave_params[f1][0]
#     res, _ = ss_vary_r0(tmp_params)

#     plt.plot(R0, res[:, [2, 3]].sum(1),
#              label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params[f1][1]}= {behave_params[f1][0]}")

# vary_params["B_fear"] = behave_params["B_fear"][0]
# vary_params["B_const"] = behave_params["B_const"][0]
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [2, 3]].sum(1),
#          label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")
# vary_params["B_social"] = 0
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [2, 3]].sum(1),
#          label=f"{behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")

# plt.text(-0.4, -0.05, caption,  va="top")
# plt.legend(loc=[1.05, 0.25])
# # plt.savefig("../img/I_by_behaviour/I_sadd_to_omega_1.png",
# # dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure()
# plt.title("Behaviour rates against disease characteristic \nfor different behaviour parameters")
# plt.xlabel("$\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
# plt.ylabel("B")

# # No behaviour
# vary_params = dict(bifurc_params)
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [1, 3, 5]].sum(1), label="No behaviour")

# var1 = "B_social"

# vary_params[var1] = behave_params[var1][0]
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [1, 3, 5]].sum(1),
#          label=f"{behave_params[var1][1]}= {behave_params[var1][0]}")


# for f1 in behave_params.keys():
#     if f1 == var1:
#         continue
#     tmp_params = dict(vary_params)
#     tmp_params[f1] = behave_params[f1][0]
#     res, _ = ss_vary_r0(tmp_params)

#     plt.plot(R0, res[:, [1, 3, 5]].sum(1),
#              label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params[f1][1]}= {behave_params[f1][0]}")

# vary_params["B_fear"] = behave_params["B_fear"][0]
# vary_params["B_const"] = behave_params["B_const"][0]
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [1, 3, 5]].sum(1),
#          label=f"{behave_params[var1][1]}= {behave_params[var1][0]}, {behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")
# vary_params["B_social"] = 0
# res, R0 = ss_vary_r0(vary_params)
# plt.plot(R0, res[:, [1, 3, 5]].sum(1),
#          label=f"{behave_params['B_fear'][1]}= {behave_params['B_fear'][0]}, {behave_params['B_const'][1]}= {behave_params['B_const'][0]}")

# plt.text(-0.4, -0.1, caption,  va="top")
# plt.legend(loc=[1.05, 0.25])
# # plt.savefig("../img/I_by_behaviour/B_sadd_to_omega_1.png",
# # dpi=600, bbox_inches="tight")
# plt.show()
