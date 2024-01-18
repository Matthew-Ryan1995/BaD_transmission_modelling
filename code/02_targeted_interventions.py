#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:10:27 2023

Explore the effects of interventions targeted at improving aspects of the Health Belief Model

@author: Matt Ryan
"""
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          # "text.usetex": True,
          "font.family": "serif"}
plt.rcParams.update(params)
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# %% Load default parameters

baseline_params = load_param_defaults()

# %% Functions

# Return the endemic disease and infection levels for given parameters


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

# Create the plot for targeted interventions


def create_plot(base_params, focus_param, save=True):

    r0_vals = np.arange(0.1, 10.1, step=0.1)

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
    # plt.title(titles[w])

    # For colours https://stats.stackexchange.com/questions/118033/best-series-of-colors-to-use-for-differentiating-series-in-publication-quality
    plt.plot(r0_vals, i_list_no_behaviour, label="No behaviour", color="black")
    plt.plot(r0_vals, i_list_base, label="BaD baseline", color="#88CCEE")
    if w in ["N_social", "N_const"]:
        plt.plot(r0_vals, i_list_10, label="Halved effect", color="#CC6677")
        plt.plot(r0_vals, i_list_90, label="Quartered effect", color="#DDCC77")
    else:
        plt.plot(r0_vals, i_list_10, label="Doubled effect", color="#CC6677")
        plt.plot(r0_vals, i_list_90, label="Quadrupled effect", color="#DDCC77")

    plt.plot([1.5, 1.5], [0, i_list_no_behaviour.max()+0.001], ":k")
    plt.text(1.6, i_list_no_behaviour.max()+0.001, "Influenza-like")
    plt.plot([8.2, 8.2], [0, i_list_no_behaviour.max()+0.001], ":k")
    plt.text(8.3, i_list_no_behaviour.max()+0.001, "Covid-like")

    plt.xlabel("Disease characteristic ($\\mathscr{R}_0^D$)")
    plt.ylabel("Endemic disease prevalence ($I^*$)")

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
    plt.savefig(f"../img/target_intervention/{titles[w]}.png",
                dpi=600, bbox_inches="tight")
    if save:
        plt.close()
    else:
        plt.show()


# %% Create plots
for w in ["B_social", "B_fear", "B_const", "N_social", "N_const"]:
    create_plot(baseline_params, w)
