#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 08:12:40 2023

@author: rya200
"""
# %% Packages

from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

# %% functions


# r0_a = np.arange(0.1, 5, step=0.1)
# r0_a_range = np.arange(0.1, 5, step=0.1)
# r0_b_range = np.arange(0.1, 3, 0.01)

# beta_vals = list()
# M3 = bad(**plot_params)
# for idx in range(len(r0_a)):

#     w = r0_a[idx]
#     ww = w * (plot_params["N_social"] +
#               plot_params["N_fear"] + plot_params["N_const"])

#     M3.update_params(**{"B_social": ww})

#     tmp = M3.Rzero()/plot_params["transmission"]

#     beta_vals.append(1/tmp)

# r0_d = np.array(beta_vals) * plot_params["infectious_period"]

# grid_range = np.meshgrid(r0_a_range, r0_b_range)

# iter_vals = np.array(grid_range).reshape(2, len(r0_a_range)*len(r0_b_range)).T
# col_vals = list()
# for idxx in range(len(iter_vals)):
#     a1 = next(i for i in range(len(r0_a)) if iter_vals[idxx, 0] <= r0_a[i])

#     b1 = r0_d[a1]

#     if (iter_vals[idxx, 1] < b1) and (iter_vals[idxx, 0] < 1):
#         col_vals.append(0)
#     elif (iter_vals[idxx, 1] < b1):
#         col_vals.append(1)
#     else:
#         col_vals.append(2)


def create_ss_region_data(input_params,
                          disease_range=[0, 5], disease_step=0.01,
                          behav_range=[0, 3], behav_step=0.01):

    r0_b = np.arange(start=behav_range[0],
                     stop=behav_range[1] + behav_step, step=behav_step)

    params = dict(input_params)
    # Normalise beta to make calculations easier
    params["transmission"] = 1

    # Find the line where R0 = 1
    M = bad(**params)
    new_betas = list()

    for idx in range(len(r0_b)):
        # Start with the R0_b value
        w = r0_b[idx]
        # Convert to w1
        ww = w * (params["N_social"] + params["N_fear"] + params["N_const"])

        M.update_params(**{"B_social": ww})

        beta_solve = M.Rzero()

        new_betas.append(1/beta_solve)

    # Calculate beta/gamma
    r0_d = np.array(new_betas) * params["infectious_period"]

    # Calculate steady state regions
    if r0_d.max() > disease_range[1]:
        disease_range[1] = r0_d.max()
    r0_d_mesh_vals = np.arange(
        start=disease_range[0], stop=disease_range[1] + disease_step, step=disease_step)

    grid_vals = np.meshgrid(r0_b, r0_d_mesh_vals)

    iter_vals = np.array(grid_vals).reshape(2, len(r0_b)*len(r0_d_mesh_vals)).T

    ss_categories = list()

    for idxx in range(len(iter_vals)):
        b_index = next(i for i in range(len(r0_b))
                       if iter_vals[idxx, 0] == r0_b[i])

        d_val = r0_d[b_index]

        # 0 - BaD free
        # 1 - B free, D endemic
        # 2 - D free, B endemic
        # 3 - full endemic

        if params["B_const"] > 0:  # Maybe need this at machine precision, not exact zero
            if iter_vals[idxx, 1] > d_val:
                ss_categories.append(3)
            else:
                ss_categories.append(2)
        elif params["B_fear"] > 0:
            if iter_vals[idxx, 1] <= d_val and iter_vals[idxx, 0] <= 1:
                ss_categories.append(0)
            elif iter_vals[idxx, 1] < d_val:
                ss_categories.append(2)
            else:
                ss_categories.append(3)
        else:
            if iter_vals[idxx, 1] <= d_val and iter_vals[idxx, 0] <= 1:
                ss_categories.append(0)
            elif iter_vals[idxx, 1] > d_val and iter_vals[idxx, 0] <= 1:
                ss_categories.append(1)
            elif iter_vals[idxx, 1] < d_val:
                ss_categories.append(2)
            else:
                ss_categories.append(3)

    ss_categories = np.array(ss_categories).reshape(grid_vals[0].shape)

    return r0_b, r0_d, grid_vals, ss_categories


def create_ss_plots(input_params, r0_b, r0_d, grid_vals, ss_categories, save=False):

    lvls = [0, 0.5, 1.5, 2.5, 3.5]
    cmap = plt.cm.RdBu_r

    if input_params["B_fear"] > 0:
        y_line = [0, 1]
    else:
        y_line = [0, grid_vals[1].max()]

    code_to_label = {
        # "transmission": "beta",
        "infectious_period": "gamma_inv",
        "immune_period": "nu_inv",
        "susc_B_efficacy": "c",
        "inf_B_efficacy": "p",
        "N_social": "a1",
        "N_fear": "a2",
        # "B_social": "w1",
        # "B_fear": "w2",
        # "B_const": "w3",
        "N_const": "a3"
    }
    code_to_latex = {
        # "transmission": "beta",
        "infectious_period": "$1/\\gamma$",
        "immune_period": "$1/\\nu$",
        "susc_B_efficacy": "$c$",
        "inf_B_efficacy": "$p$",
        "N_social": "$\\alpha_1$",
        "N_fear": "$\\alpha_2$",
        # "B_social": "w1",
        # "B_fear": "w2",
        # "B_const": "w3",
        "N_const": "$\\alpha_3$"
    }
    title = "Steady state regions: "
    if np.isclose(input_params["B_fear"], 0):
        title += "$\\omega_2$ = 0, "
    if np.isclose(input_params["B_const"], 0):
        title += "$\\omega_3$ = 0"
    title += "\n| "

    for var in code_to_latex.keys():
        if var == "N_social":
            title += "\n| "
        title += code_to_latex[var] + " = " + \
            str(np.round(input_params[var], 1)) + " | "
    if input_params["B_fear"] > 0:
        title += "$\\omega_2$  = " + \
            str(np.round(input_params["B_fear"], 1)) + " | "
    if input_params["B_const"] > 0:
        title += "$\\omega_2$  = " + \
            str(np.round(input_params["B_const"], 1)) + " | "

    save_lbl = "steady_state_regions/ss_regions_"
    for var in code_to_label.keys():
        save_lbl += code_to_label[var] + "_" + str(input_params[var]) + "_"

    append_txt = ""
    if input_params["B_fear"] > 0:
        save_lbl += "w2_" + str(np.round(input_params["B_fear"], 1)) + "_"
    else:
        append_txt += "w2_0_"
    if input_params["B_const"] > 0:
        save_lbl += "w3_" + str(np.round(input_params["B_const"], 1))
    else:
        append_txt += "w3_0"

    caption_txt = "Dark blue - BaD free; Light blue - B free, D endemic; \nLight red - B endemic, D free; Dark red - BaD endemic"

    plt.figure()
    plt.title(title)
    plt.tight_layout()
    plt.contourf(grid_vals[0], grid_vals[1], ss_categories,
                 levels=lvls,  cmap=cmap)
    plt.plot(r0_b, r0_d, linestyle="-", color="black", linewidth=2)
    if np.isclose(input_params["B_const"], 0):
        plt.plot([1, 1], y_line, color="black", linewidth=2)
    plt.xlabel(
        "Behavioural characteristic ($\\omega_1 / (\\alpha_1 + \\alpha_2 + \\alpha_3)$)")
    plt.ylabel(
        "Epidemic characteristic ($\\beta/\\gamma$)")

    y_ticks = plt.yticks()[0]
    y_spacing = y_ticks[1] - y_ticks[0]

    plt.text(0., -y_spacing - 0.5, caption_txt, va="top")
    if save:
        plt.savefig("../img/" + save_lbl + append_txt +
                    ".png", dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# %%


plot_params_baseline = dict()
plot_params_baseline["transmission"] = 1
plot_params_baseline["infectious_period"] = 1/1
plot_params_baseline["immune_period"] = 1/0.5
plot_params_baseline["av_lifespan"] = 0  # Turning off demography
plot_params_baseline["susc_B_efficacy"] = 0.5
plot_params_baseline["inf_B_efficacy"] = 0.3
plot_params_baseline["N_social"] = 0.2
plot_params_baseline["N_fear"] = 1.1
plot_params_baseline["B_social"] = 1.3
plot_params_baseline["B_fear"] = 0.5
plot_params_baseline["B_const"] = 0.7
plot_params_baseline["N_const"] = 0.9

for c in [0.2, 0.4, 0.8]:
    for p in [0.0, 0.4, 0.6]:
        plot_params = dict(plot_params_baseline)
        plot_params["susc_B_efficacy"] = c
        plot_params["inf_B_efficacy"] = p

        r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
            plot_params)
        create_ss_plots(plot_params, r0_b, r0_d,
                        grid_vals, ss_categories, save=True)

        plot_params["B_const"] = 0
        r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
            plot_params)
        create_ss_plots(plot_params, r0_b, r0_d,
                        grid_vals, ss_categories, save=True)

        plot_params["B_fear"] = 0
        r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
            plot_params)
        create_ss_plots(plot_params, r0_b, r0_d,
                        grid_vals, ss_categories, save=True)


# %%
# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params_baseline)

# # %%

# create_ss_plots(plot_params_baseline, r0_b, r0_d,
#                 grid_vals, ss_categories, save=False)
