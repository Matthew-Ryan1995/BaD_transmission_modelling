#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:10:27 2023

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

m_params = load_param_defaults()
# m_params["transmission"] = 1
# m_params["infectious_period"] = 1/1
# m_params["immune_period"] = 1/0.4
# m_params["av_lifespan"] = 0  # Turning off demography
# m_params["susc_B_efficacy"] = 0.5
# m_params["inf_B_efficacy"] = 0.5
# m_params["N_social"] = 1.25  # 0.2
# m_params["B_social"] = 0.4  # 1.3
# m_params["B_fear"] = 8.  # 0.5
# m_params["B_const"] = 0.2  # 0.7
# m_params["N_const"] = 0.6  # 0.9
# m_params = dict()
# m_params["transmission"] = 1
# m_params["infectious_period"] = 1/1
# m_params["immune_period"] = 1/0.5
# m_params["av_lifespan"] = 0  # Turning off demography
# m_params["susc_B_efficacy"] = 0.5
# m_params["inf_B_efficacy"] = 0.3
# m_params["N_social"] = 1  # 0.2
# m_params["B_social"] = 0.1  # 1.3
# m_params["B_fear"] = 0.1  # 0.5
# m_params["B_const"] = 0.1  # 0.7
# m_params["N_const"] = 0.5  # 0.9

# %%

r0_vals = np.arange(0.1, 5.1, step=0.1)

w1_vals = np.arange(0.5, 3.5, step=0.5)


plt.figure()
for w1 in w1_vals:
    v_params = dict(m_params)
    v_params["B_const"] = w1
    i_list = list()
    tmp_M = bad(**v_params)
    # multi_val = tmp_M.Rzero()
    for r0 in r0_vals:
        v_params["transmission"] = r0 / v_params["infectious_period"]
        # v_params["transmission"] = r0 / multi_val

        ss, _ = find_ss(v_params)

        i_list.append(ss[[2, 3]].sum())
    i_list = np.array(i_list)

    plt.plot(r0_vals, i_list, label=f"w1 = {w1}")

plt.legend()
plt.show()

# %%


def return_i_b(params):
    v_params = dict(params)
    r0_vals = np.arange(0.1, 5.1, step=0.1)
    i_list = list()
    b_list = list()
    for r0 in r0_vals:
        v_params["transmission"] = r0 / v_params["infectious_period"]
        ss, _ = find_ss(v_params)

        i_list.append(ss[[2, 3]].sum())
        b_list.append(ss[[1, 3, 5]].sum())
    i_list = np.array(i_list)
    b_list = np.array(b_list)

    return i_list, b_list


r0_vals = np.arange(0.1, 5.1, step=0.1)
w1_vals = np.arange(0.5, 2., step=0.5)

v_params = dict(m_params)

i_list_base, b_list_base = return_i_b(v_params)

v_params = dict(m_params)
v_params["B_social"] = w1_vals[-1]
i_list_w1, b_list_w1 = return_i_b(v_params)

v_params = dict(m_params)
v_params["B_fear"] = w1_vals[-1]
i_list_w2, b_list_w2 = return_i_b(v_params)

v_params = dict(m_params)
v_params["B_const"] = w1_vals[-1]
i_list_w3, b_list_w3 = return_i_b(v_params)

# plt.plot(r0_vals, i_list, label=f"w1 = {w1}")

# %%

plt.figure()
plt.title("Varying w1")
plt.fill_between(r0_vals, i_list_base, i_list_w1, color="green", alpha=0.1)
plt.plot(r0_vals, i_list_base, color="black", label="Baseline")
plt.plot(r0_vals, i_list_w1, color="green", label="Increasing w1")
plt.xlabel("$\\beta/\\gamma$")
plt.ylabel("Endemic disease prevalence")
plt.legend()
plt.show()

plt.figure()
plt.title("Varying w2")
plt.fill_between(r0_vals, i_list_base, i_list_w2, color="blue", alpha=0.1)
plt.plot(r0_vals, i_list_base, color="black", label="Baseline")
plt.plot(r0_vals, i_list_w2, color="blue", label="Increasing w2")
plt.xlabel("$\\beta/\\gamma$")
plt.ylabel("Endemic disease prevalence")
plt.legend()
plt.show()

plt.figure()
plt.title("Varying w3")
plt.fill_between(r0_vals, i_list_base, i_list_w3, color="red", alpha=0.1)
plt.plot(r0_vals, i_list_base, color="black", label="Baseline")
plt.plot(r0_vals, i_list_w3, color="red", label="Increasing w3")
plt.xlabel("$\\beta/\\gamma$")
plt.ylabel("Endemic disease prevalence")
plt.legend()
plt.show()

plt.figure()
plt.title("Varying w1")
plt.fill_between(r0_vals, b_list_base, b_list_w1, color="green", alpha=0.1)
plt.plot(r0_vals, b_list_base, color="black", label="Baseline")
plt.plot(r0_vals, b_list_w1, color="green", label="Increasing w1")
plt.xlabel("$\\beta/\\gamma$")
plt.ylabel("Endemic behaviour prevalence")
plt.legend()
plt.show()

plt.figure()
plt.title("Varying w2")
plt.fill_between(r0_vals, b_list_base, b_list_w2, color="blue", alpha=0.1)
plt.plot(r0_vals, b_list_base, color="black", label="Baseline")
plt.plot(r0_vals, b_list_w2, color="blue", label="Increasing w2")
plt.xlabel("$\\beta/\\gamma$")
plt.ylabel("Endemic behaviour prevalence")
plt.legend()
plt.show()

plt.figure()
plt.title("Varying w3")
plt.fill_between(r0_vals, b_list_base, b_list_w3, color="red", alpha=0.1)
plt.plot(r0_vals, b_list_base, color="black", label="Baseline")
plt.plot(r0_vals, b_list_w3, color="red", label="Increasing w3")
plt.xlabel("$\\beta/\\gamma$")
plt.ylabel("Endemic behaviour prevalence")
plt.legend()
plt.show()

# %%

baseline_params = load_param_defaults()
# baseline_params["transmission"] = 1
# baseline_params["infectious_period"] = 1
# baseline_params["immune_period"] = 1/0.4
# baseline_params["av_lifespan"] = 0  # Turning off demography
# baseline_params["susc_B_efficacy"] = 0.5
# baseline_params["inf_B_efficacy"] = 0.5
# baseline_params["N_social"] = 1.25
# baseline_params["B_social"] = 0.4
# baseline_params["B_fear"] = 0.4/0.05
# baseline_params["B_const"] = 0.2
# baseline_params["N_const"] = 0.6
# baseline_params = dict()
# baseline_params["transmission"] = 1
# baseline_params["infectious_period"] = 1
# baseline_params["immune_period"] = 2
# baseline_params["av_lifespan"] = 0  # Turning off demography
# baseline_params["susc_B_efficacy"] = 0.3
# baseline_params["inf_B_efficacy"] = 0.3
# baseline_params["N_social"] = 0.2
# baseline_params["B_social"] = 0.5
# baseline_params["B_fear"] = 0.5
# baseline_params["B_const"] = 0.5
# baseline_params["N_const"] = 0.9

r0_vals = np.arange(0.1, 8.1, step=0.1)


def return_i_b(params, R0):
    v_params = dict(params)
    r0_vals = R0
    i_list = list()
    b_list = list()
    for r0 in r0_vals:
        v_params["transmission"] = r0 / v_params["infectious_period"]
        ss, _ = find_ss(v_params)

        i_list.append(ss[[2, 3]].sum())
        b_list.append(ss[[1, 3, 5]].sum())
    i_list = np.array(i_list)
    b_list = np.array(b_list)

    return i_list, b_list


i_list_base, bb_1 = return_i_b(baseline_params, r0_vals)

no_behaviour = dict(baseline_params)
no_behaviour["susc_B_efficacy"] = 0.0
no_behaviour["inf_B_efficacy"] = 0.0

i_list_no_behaviour, _ = return_i_b(no_behaviour, r0_vals)

param_vary = ["B_social", "B_fear", "B_const"]

titles = {
    "B_social": "Social cues to action",
    "B_fear": "Perception of illness threat",
    "B_const": "Self efficacy and perceived benefits"
}

for w in param_vary:
    increase_10 = dict(baseline_params)
    increase_10[w] = 2 * increase_10[w]

    i_list_10, _ = return_i_b(increase_10, r0_vals)

    increase_90 = dict(baseline_params)
    increase_90[w] = 4 * increase_90[w]

    i_list_90, bb = return_i_b(increase_90, r0_vals)

    plt.figure()
    # plt.title(titles[w])

    # For colours https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
    plt.plot(r0_vals, i_list_no_behaviour, label="No behaviour", color="black")
    plt.plot(r0_vals, i_list_base, label="BaD baseline", color="#88CCEE")
    plt.plot(r0_vals, i_list_10, label="Doubled effect", color="#CC6677")
    plt.plot(r0_vals, i_list_90, label="Quadrupled effect", color="#DDCC77")

    plt.plot([1.4, 1.4], [0, 0.3], ":k")
    plt.text(1.5, 0.25, "Flu-like")
    plt.plot([5.4, 5.4], [0, 0.3], ":k")
    plt.text(5.5, 0.1, "Covid-like")

    plt.xlabel("Disease transmissibility ($\\mathscr{R}_0^D$)")
    plt.ylabel("Long-term disease prevalence")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    plt.savefig(
        f"../img/convergence/{titles[w]}.png", dpi=600, bbox_inches="tight")
    plt.show()

# %%


def create_plot(base_params, focus_param):

    baseline_params = dict(base_params)

    i_list_base, bb_1 = return_i_b(baseline_params, r0_vals)

    no_behaviour = dict(baseline_params)
    no_behaviour["susc_B_efficacy"] = 0.0
    no_behaviour["inf_B_efficacy"] = 0.0

    i_list_no_behaviour, _ = return_i_b(no_behaviour, r0_vals)

    param_vary = ["B_social", "B_fear", "B_const"]

    titles = {
        "B_social": "Social cues to action",
        "B_fear": "Perception of illness threat",
        "B_const": "Self efficacy and perceived benefits",
        "N_social": "Social cues for abandonment",
        "N_const": "Internal cues and perceived barriers"
    }

    w = focus_param
    if w in ["N_social", "N_const"]:
        increase_10 = dict(baseline_params)
        increase_10[w] = increase_10[w]/2

        i_list_10, _ = return_i_b(increase_10, r0_vals)

        increase_90 = dict(baseline_params)
        increase_90[w] = increase_90[w]/4

        i_list_90, bb = return_i_b(increase_90, r0_vals)
    else:
        increase_10 = dict(baseline_params)
        increase_10[w] = 2 * increase_10[w]

        i_list_10, _ = return_i_b(increase_10, r0_vals)

        increase_90 = dict(baseline_params)
        increase_90[w] = 4 * increase_90[w]

        i_list_90, bb = return_i_b(increase_90, r0_vals)

    plt.figure()
    plt.title(titles[w])

    # For colours https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
    plt.plot(r0_vals, i_list_no_behaviour, label="No behaviour", color="black")
    plt.plot(r0_vals, i_list_base, label="BaD baseline", color="#88CCEE")
    if w in ["N_social", "N_const"]:
        plt.plot(r0_vals, i_list_10, label="Halved effect", color="#CC6677")
        plt.plot(r0_vals, i_list_90, label="Quartered effect", color="#DDCC77")
    else:
        plt.plot(r0_vals, i_list_10, label="Doubled effect", color="#CC6677")
        plt.plot(r0_vals, i_list_90, label="Quadrupled effect", color="#DDCC77")

    plt.plot([1.4, 1.4], [0, i_list_no_behaviour.max()], ":k")
    plt.text(1.5, i_list_no_behaviour.max(), "Flu-like")
    plt.plot([5.4, 5.4], [0, i_list_no_behaviour.max()], ":k")
    plt.text(5.5, i_list_no_behaviour.max(), "Covid-like")

    plt.xlabel("Disease characteristic ($\\mathscr{R}_0^D$)")
    plt.ylabel("Endemic disease prevalence ($I^*$)")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    plt.savefig(f"../img/target_intervention/{titles[w]}.png",
                dpi=600, bbox_inches="tight")
    plt.show()


def run_model(params, R0):
    tmp_params = dict(params)
    tmp_params["transmission"] = R0/tmp_params["infectious_period"]

    Sb = 1e-3
    In = 1e-3
    Ib = Rb = 0
    Rn = 0.2

    Sn = 1 - Sb - In - Ib - Rb - Rn

    IC = [Sn, Sb, In, Ib, Rn, Rb]

    t_start = 0
    t_end = 1000

    M = bad(**tmp_params)
    M.run(IC, t_start, t_end)

    return M.get_I(), M.get_S()


def create_phase_plot(base_params, focus_param, R0d=5.4):

    baseline_params = dict(base_params)
    baseline_params["infectious_period"] = 7
    baseline_params["immune_period"] = 240

    i_base, s_base = run_model(baseline_params, R0d)

    no_behaviour = dict(baseline_params)
    no_behaviour["susc_B_efficacy"] = 0.0
    no_behaviour["inf_B_efficacy"] = 0.0

    i_no_behaviour, s_no_behaviour = run_model(no_behaviour, R0d)

    param_vary = ["B_social", "B_fear", "B_const"]

    titles = {
        "B_social": "Social cues to action",
        "B_fear": "Perception of illness threat",
        "B_const": "Self efficacy and perceived benefits",
        "N_social": "Social cues for abandonment",
        "N_const": "Internal cues and perceived barriers"
    }

    w = focus_param
    if w in ["N_social", "N_const"]:
        increase_2 = dict(baseline_params)
        increase_2[w] = increase_2[w]/2

        i_2, s_2 = run_model(increase_2, R0d)

        increase_4 = dict(baseline_params)
        increase_4[w] = increase_4[w]/4

        i_4, s_4 = run_model(increase_4, R0d)
    else:
        increase_2 = dict(baseline_params)
        increase_2[w] = 2 * increase_2[w]

        i_2, s_2 = run_model(increase_2, R0d)

        increase_4 = dict(baseline_params)
        increase_4[w] = 4 * increase_4[w]

        i_4, s_4 = run_model(increase_4, R0d)

    plt.figure()
    plt.title(titles[w] + ", $\\mathscr{R}_0^D=$" + f"{R0d}")

    # For colours https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
    plt.plot(s_no_behaviour, i_no_behaviour,
             label="No behaviour", color="black")
    plt.plot(s_base, i_base, label="BaD baseline", color="#88CCEE")
    if w in ["N_social", "N_const"]:
        plt.plot(s_2, i_2, label="Halved effect", color="#CC6677")
        plt.plot(s_4, i_4, label="Quartered effect", color="#DDCC77")
    else:
        plt.plot(s_2, i_2, label="Doubled effect", color="#CC6677")
        plt.plot(s_4, i_4, label="Quadrupled effect", color="#DDCC77")

    plt.xlabel("S")
    plt.ylabel("I")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    plt.savefig(f"../img/target_intervention/phase_plot_{R0d}_{titles[w]}.png",
                dpi=600, bbox_inches="tight")
    plt.show()


for w in ["B_social", "B_fear", "B_const", "N_social", "N_const"]:
    create_plot(baseline_params, w)
    create_phase_plot(baseline_params, w)
    create_phase_plot(baseline_params, w, R0d=1.4)

# %%

s, i = run_model(baseline_params, 5.4)
