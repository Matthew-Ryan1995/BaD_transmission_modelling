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
#     ww = w * (plot_params["N_social"] + plot_params["N_const"])

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


# This version of the code does the slow, but more correct calculation.  Actually calculates the SS for each region
# def create_ss_region_data(input_params,
#                           disease_range=[0, 5], disease_step=0.01,
#                           behav_range=[0, 3], behav_step=0.01):

#     r0_b = np.arange(start=behav_range[0],
#                      stop=behav_range[1] + behav_step, step=behav_step)

#     params = dict(input_params)
#     # Normalise beta to make calculations easier
#     params["transmission"] = 1

#     # Find the line where R0 = 1
#     M = bad(**params)
#     new_betas = list()

#     for idx in range(len(r0_b)):
#         # Start with the R0_b value
#         w = r0_b[idx]
#         # Convert to w1
#         ww = w * (params["N_social"] + params["N_const"])

#         M.update_params(**{"B_social": ww})

#         beta_solve = M.Rzero()

#         new_betas.append(1/beta_solve)

#     # Calculate beta/gamma
#     r0_d = np.array(new_betas) * params["infectious_period"]

#     # Calculate steady state regions
#     if r0_d.max() > disease_range[1]:
#         disease_range[1] = r0_d.max()
#     r0_d_mesh_vals = np.arange(
#         start=disease_range[0], stop=disease_range[1] + disease_step, step=disease_step)

#     grid_vals = np.meshgrid(r0_b, r0_d_mesh_vals)

#     iter_vals = np.array(grid_vals).reshape(2, len(r0_b)*len(r0_d_mesh_vals)).T

#     ss_categories = list()

#     for idxx in range(len(iter_vals)):
#         b_index = next(i for i in range(len(r0_b))
#                        if iter_vals[idxx, 0] == r0_b[i])

#         d_val = r0_d[b_index]

#         params["transmission"] = iter_vals[idxx, 1] / \
#             params["infectious_period"]
#         params["B_social"] = iter_vals[idxx, 0] * \
#             (params["N_social"] + params["N_const"])

#         ss, _ = find_ss(params)

#         ss.round(7)

#         I = ss[[2, 3]].sum()
#         B = ss[[1, 3, 5]].sum()

#         if np.isclose(I, 0):
#             if np.isclose(B, 0):
#                 ss_categories.append(0)
#             else:
#                 ss_categories.append(2)
#         else:
#             if np.isclose(B, 0):
#                 ss_categories.append(1)
#             else:
#                 ss_categories.append(3)

#         # 0 - BaD free
#         # 1 - B free, D endemic
#         # 2 - D free, B endemic
#         # 3 - full endemic

#         # if params["B_const"] > 0:  # Maybe need this at machine precision, not exact zero
#         #     if iter_vals[idxx, 1] > d_val:
#         #         ss_categories.append(3)
#         #     else:
#         #         ss_categories.append(2)
#         # elif params["B_fear"] > 0:
#         #     if iter_vals[idxx, 1] <= d_val and iter_vals[idxx, 0] <= 1:
#         #         ss_categories.append(0)
#         #     elif iter_vals[idxx, 1] < d_val:
#         #         ss_categories.append(2)
#         #     else:
#         #         ss_categories.append(3)
#         # else:
#         #     if iter_vals[idxx, 1] <= d_val and iter_vals[idxx, 0] <= 1:
#         #         ss_categories.append(0)
#         #     elif iter_vals[idxx, 1] > d_val and iter_vals[idxx, 0] <= 1:
#         #         ss_categories.append(1)
#         #     elif iter_vals[idxx, 1] < d_val:
#         #         ss_categories.append(2)
#         #     else:
#         #         ss_categories.append(3)

#     ss_categories = np.array(ss_categories).reshape(grid_vals[0].shape)

#     return r0_b, r0_d, grid_vals, ss_categories

# This version of the code does to quick checks to match what the slow version does.
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
        ww = w * (params["N_social"] + params["N_const"])

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


# def create_ss_plots(input_params, r0_b, r0_d, grid_vals, ss_categories, save=False):

#     lvls = [0, 0.5, 1.5, 2.5, 3.5]
#     cmap = plt.cm.RdBu_r

#     if input_params["B_fear"] > 0:
#         y_line = [0, 1]
#     else:
#         y_line = [0, grid_vals[1].max()]

#     code_to_label = {
#         # "transmission": "beta",
#         "infectious_period": "gamma_inv",
#         "immune_period": "nu_inv",
#         "susc_B_efficacy": "c",
#         "inf_B_efficacy": "p",
#         "N_social": "a1",
#         # "B_social": "w1",
#         # "B_fear": "w2",
#         # "B_const": "w3",
#         "N_const": "a3"
#     }
#     code_to_latex = {
#         # "transmission": "beta",
#         "infectious_period": "$1/\\gamma$",
#         "immune_period": "$1/\\nu$",
#         "susc_B_efficacy": "$c$",
#         "inf_B_efficacy": "$p$",
#         "N_social": "$\\alpha_1$",
#         # "B_social": "w1",
#         # "B_fear": "w2",
#         # "B_const": "w3",
#         "N_const": "$\\alpha_3$"
#     }
#     title = "Steady state regions: "
#     if np.isclose(input_params["B_fear"], 0):
#         title += "$\\omega_2$ = 0, "
#     if np.isclose(input_params["B_const"], 0):
#         title += "$\\omega_3$ = 0"
#     title += "\n| "

#     for var in code_to_latex.keys():
#         if var == "N_social":
#             title += "\n| "
#         title += code_to_latex[var] + " = " + \
#             str(np.round(input_params[var], 1)) + " | "
#     if input_params["B_fear"] > 0:
#         title += "$\\omega_2$  = " + \
#             str(np.round(input_params["B_fear"], 1)) + " | "
#     if input_params["B_const"] > 0:
#         title += "$\\omega_3$  = " + \
#             str(np.round(input_params["B_const"], 1)) + " | "

#     save_lbl = "steady_state_regions/ss_regions_"
#     for var in code_to_label.keys():
#         save_lbl += code_to_label[var] + "_" + str(input_params[var]) + "_"

#     append_txt = ""
#     if input_params["B_fear"] > 0:
#         save_lbl += "w2_" + str(np.round(input_params["B_fear"], 1)) + "_"
#     else:
#         append_txt += "w2_0_"
#     if input_params["B_const"] > 0:
#         save_lbl += "w3_" + str(np.round(input_params["B_const"], 1))
#     else:
#         append_txt += "w3_0"

#     caption_txt = "Dark blue - BaD free; Light blue - B free, D endemic; \nLight red - B endemic, D free; Dark red - BaD endemic"

#     plt.figure()
#     plt.title(title)
#     plt.tight_layout()
#     plt.contourf(grid_vals[0], grid_vals[1], ss_categories,
#                  levels=lvls,  cmap=cmap)
#     plt.plot(r0_b, r0_d, linestyle="-", color="black", linewidth=2)
#     if np.isclose(input_params["B_const"], 0):
#         plt.plot([1, 1], y_line, color="black", linewidth=2)
#     plt.xlabel(
#         "Behavioural characteristic ($\\omega_1 / (\\alpha_1  + \\alpha_2)$)")
#     plt.ylabel(
#         "Epidemic characteristic ($\\beta/\\gamma$)")

#     y_ticks = plt.yticks()[0]
#     y_spacing = y_ticks[1] - y_ticks[0]

#     plt.text(0., -y_spacing - 0.5, caption_txt, va="top")
#     if save:
#         plt.savefig("../img/" + save_lbl + append_txt +
#                     ".png", dpi=600, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()


def create_ss_plots_2(input_params, r0_b, r0_d, grid_vals, ss_categories, save=False):

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
    plt.contourf(grid_vals[1], grid_vals[0], ss_categories,
                 levels=lvls,  cmap=cmap)
    plt.plot(r0_d, r0_b, linestyle="-", color="black", linewidth=2)
    if np.isclose(input_params["B_const"], 0):
        plt.plot(y_line, [1, 1], color="black", linewidth=2)
    plt.ylabel(
        # "Behavioural characteristic ($\\omega_1 / (\\alpha_1 + \\alpha_2 + \\alpha_3)$)"
        "$\\mathscr{R}_0^{B}$"
    )
    plt.xlabel(
        # "Epidemic characteristic ($\\beta/\\gamma$)"
        "$\\mathscr{R}_0^{D}$"
    )

    y_ticks = plt.yticks()[0]
    y_spacing = y_ticks[1] - y_ticks[0]

    # plt.text(0., -y_spacing - 0.5, caption_txt, va="top")
    x_pos = 1.5+(grid_vals[1].max() - 1)/2
    y_pos = 1+(grid_vals[0].max() - 1)/2
    if np.isclose(input_params["B_const"], 0):
        plt.text(0.5, 0.5, "$E_{00}$", size=16, va="center", ha="center")
        if np.isclose(input_params["B_fear"], 0):
            plt.text(x_pos, 0.5, "$E_{0D}$",
                     size=16, va="center", ha="center")

    plt.text(0.5, y_pos, "$E_{B0}$", size=16, va="center", ha="center")
    if np.isclose(input_params["B_fear"], 0):
        plt.text(x_pos, y_pos, "$E_{BD}$",
                 size=16, va="center", ha="center")
    else:
        plt.text(x_pos, y_pos-1, "$E_{BD}$",
                 size=16, va="center", ha="center")

    if save:
        plt.savefig("../img/" + save_lbl + append_txt +
                    ".png", dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_ss_plots_convergence(input_params, r0_b, r0_d, grid_vals, ss_categories, save=False):

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

    save_lbl = "convergence/ss_regions_"
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
    # plt.title(title)
    plt.tight_layout()
    plt.contourf(grid_vals[1], grid_vals[0], ss_categories,
                 levels=lvls,  cmap=cmap)
    plt.plot(r0_d, r0_b, linestyle="-", color="black", linewidth=2)
    if np.isclose(input_params["B_const"], 0):
        plt.plot(y_line, [1, 1], color="black", linewidth=2)
    plt.ylabel(
        # "Behavioural characteristic ($\\omega_1 / (\\alpha_1 + \\alpha_2 + \\alpha_3)$)"
        "Social 'transmisibility' of behaviour ($\\mathscr{R}_0^{B}$)"
    )
    plt.xlabel(
        # "Epidemic characteristic ($\\beta/\\gamma$)"
        "Disease transmisibility ($\\mathscr{R}_0^{D}$)"
    )

    y_ticks = plt.yticks()[0]
    y_spacing = y_ticks[1] - y_ticks[0]

    # plt.text(0., -y_spacing - 0.5, caption_txt, va="top")
    x_pos = 1.5+(grid_vals[1].max() - 1)/2
    y_pos = 1+(grid_vals[0].max() - 1)/2
    if np.isclose(input_params["B_const"], 0):
        plt.text(0.5, 0.5, "Neither", size=16, va="center", ha="center")
        if np.isclose(input_params["B_fear"], 0):
            plt.text(x_pos, 0.5, "No behaviour,\ndisease present",
                     size=16, va="center", ha="center")

    plt.text(1.5, y_pos+0.5, "Behaviour present,\nno disease",
             size=16, va="center", ha="center")
    if np.isclose(input_params["B_fear"], 0):
        plt.text(x_pos+0.2, y_pos, "Behaviour and\ndisease present",
                 size=16, va="center", ha="center")
    else:
        plt.text(x_pos, y_pos-1, "Behaviour and\ndisease present",
                 size=16, va="center", ha="center")

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
plot_params_baseline["immune_period"] = 1/0.4
plot_params_baseline["av_lifespan"] = 0  # Turning off demography
plot_params_baseline["susc_B_efficacy"] = 0.5
plot_params_baseline["inf_B_efficacy"] = 0.5
plot_params_baseline["N_social"] = 1.25
plot_params_baseline["B_social"] = 0.4
plot_params_baseline["B_fear"] = 8.0
plot_params_baseline["B_const"] = 0.2
plot_params_baseline["N_const"] = 0.6
# plot_params_baseline = dict()
# plot_params_baseline["transmission"] = 1
# plot_params_baseline["infectious_period"] = 1/1
# plot_params_baseline["immune_period"] = 1/0.5
# plot_params_baseline["av_lifespan"] = 0  # Turning off demography
# plot_params_baseline["susc_B_efficacy"] = 0.5
# plot_params_baseline["inf_B_efficacy"] = 0.3
# plot_params_baseline["N_social"] = 0.2
# plot_params_baseline["B_social"] = 1.3
# plot_params_baseline["B_fear"] = 0.5
# plot_params_baseline["B_const"] = 0.7
# plot_params_baseline["N_const"] = 0.9

# %%

# for c in [0.8]:  # [0.2, 0.4, 0.8]:
#     for p in [0.6]:  # [0.0, 0.4, 0.6]:
plot_params = dict(plot_params_baseline)
# plot_params["susc_B_efficacy"] = c
# plot_params["inf_B_efficacy"] = p

r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
    plot_params, disease_step=0.01, behav_step=0.01)
create_ss_plots_2(plot_params, r0_b, r0_d,
                  grid_vals, ss_categories, save=True)
create_ss_plots_convergence(plot_params, r0_b, r0_d,
                            grid_vals, ss_categories, save=True)

plot_params["B_const"] = 0
r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
    plot_params, disease_step=0.01, behav_step=0.01)
create_ss_plots_2(plot_params, r0_b, r0_d,
                  grid_vals, ss_categories, save=True)
create_ss_plots_convergence(plot_params, r0_b, r0_d,
                            grid_vals, ss_categories, save=True)

plot_params["B_fear"] = 0
r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
    plot_params, disease_step=0.01, behav_step=0.01)
create_ss_plots_2(plot_params, r0_b, r0_d,
                  grid_vals, ss_categories, save=True)
create_ss_plots_convergence(plot_params, r0_b, r0_d,
                            grid_vals, ss_categories, save=True)

# %% Check 1: looking at all combos of omegas.
# Results: What we expect.  Only cases to consider are w3 on, w3 off w2 on, w3 and w2 off
# w1-w2-w3
# w1-w2, w1-w3
# w2-w3
# w1, w2,w3
# plot_params = dict(plot_params_baseline)
# plot_params["susc_B_efficacy"] = 0.8
# plot_params["inf_B_efficacy"] = 0.6


# # w1-w2-w3
# plot_params["B_social"] = 1.3
# plot_params["B_fear"] = 0.5
# plot_params["B_const"] = 0.7

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# # w1-w2
# plot_params["B_social"] = 1.3
# plot_params["B_fear"] = 0.5
# plot_params["B_const"] = 0.

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# # w1-w3
# plot_params["B_social"] = 1.3
# plot_params["B_fear"] = 0.
# plot_params["B_const"] = 0.7

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# # w2-w3
# plot_params["B_social"] = 0
# plot_params["B_fear"] = 0.5
# plot_params["B_const"] = 0.7

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# # w1
# plot_params["B_social"] = 1.3
# plot_params["B_fear"] = 0.
# plot_params["B_const"] = 0.

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# # w2
# plot_params["B_social"] = 0.
# plot_params["B_fear"] = 0.5
# plot_params["B_const"] = 0.

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# # w3
# plot_params["B_social"] = 1.3
# plot_params["B_fear"] = 0.5
# plot_params["B_const"] = 0.7

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# %% Check 2: extreme values
# p and c done: No extreme differences
# gamma betwen 4 and 14
# nu between 90 and 360
# a1 between 0 and 1
# a2 between 0 and 1 - This one looks weird, but whena2=0 it just takes forever to actually see infection.


# # gamma
# plot_params = dict(plot_params_baseline)
# plot_params["susc_B_efficacy"] = 0.8
# plot_params["inf_B_efficacy"] = 0.6

# plot_params["infectious_period"] = 4

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)

# plot_params["infectious_period"] = 14

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)

# # nu
# plot_params = dict(plot_params_baseline)
# plot_params["susc_B_efficacy"] = 0.8
# plot_params["inf_B_efficacy"] = 0.6

# plot_params["immune_period"] = 90

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)

# plot_params["immune_period"] = 360

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)

# # a1
# plot_params = dict(plot_params_baseline)
# plot_params["susc_B_efficacy"] = 0.8
# plot_params["inf_B_efficacy"] = 0.6

# plot_params["N_social"] = 0

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)

# plot_params["N_social"] = 1

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)
# # a2
# plot_params = dict(plot_params_baseline)
# plot_params["susc_B_efficacy"] = 0.8
# plot_params["inf_B_efficacy"] = 0.6

# plot_params["N_const"] = 0

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)

# plot_params["N_const"] = 1

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params, disease_step=0.01, behav_step=0.01)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=True)


# %%
# w1 = 8
# R0_d = 1.4
# R0_b = 1.01
# gamma = 1/7


# cust_params = dict()
# cust_params["transmission"] = R0_d*gamma
# cust_params["infectious_period"] = 1/gamma
# cust_params["immune_period"] = 240
# cust_params["av_lifespan"] = 0  # Turning off demography
# cust_params["susc_B_efficacy"] = 0.1
# cust_params["inf_B_efficacy"] = 0.1
# cust_params["N_social"] = 0.5
# cust_params["N_const"] = 0.01
# cust_params["B_social"] = R0_b * \
#     (cust_params["N_social"]  + cust_params["N_const"])
# cust_params["B_fear"] = 0.01  # w1
# cust_params["B_const"] = 0.01


# plot_params = dict(cust_params)
# # plot_params["susc_B_efficacy"] = 0.3
# # plot_params["inf_B_efficacy"] = 0.3

# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# plot_params["B_const"] = 0
# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# plot_params["B_fear"] = 0
# r0_b, r0_d, grid_vals, ss_categories = create_ss_region_data(
#     plot_params)
# create_ss_plots_2(plot_params, r0_b, r0_d,
#                   grid_vals, ss_categories, save=False)

# %%
# params = dict(plot_params_baseline)

# behav_range = [0, 2]
# disease_range = [0, 3]
# behav_step = 0.01
# disease_step = 0.01

# r0_b = np.arange(start=behav_range[0],
#                  stop=behav_range[1] + behav_step, step=behav_step)

# # Normalise beta to make calculations easier
# params["transmission"] = 1

# # Find the line where R0 = 1
# M = bad(**params)
# new_betas = list()

# for idx in range(len(r0_b)):
#     # Start with the R0_b value
#     w = r0_b[idx]
#     # Convert to w1
#     ww = w * (params["N_social"] + params["N_const"])

#     M.update_params(**{"B_social": ww})

#     beta_solve = M.Rzero()

#     new_betas.append(1/beta_solve)

# # Calculate beta/gamma
# r0_d = np.array(new_betas) * params["infectious_period"]

# # Calculate steady state regions
# if r0_d.max() > disease_range[1]:
#     disease_range[1] = r0_d.max()
# r0_d_mesh_vals = np.arange(
#     start=disease_range[0], stop=disease_range[1] + disease_step, step=disease_step)

# grid_vals = np.meshgrid(r0_b, r0_d_mesh_vals)

# iter_vals = np.array(grid_vals).reshape(2, len(r0_b)*len(r0_d_mesh_vals)).T

# ss_categories = list()

# for idxx in range(len(iter_vals)):
#     b_index = next(i for i in range(len(r0_b))
#                    if iter_vals[idxx, 0] == r0_b[i])

#     d_val = r0_d[b_index]

#     params["transmission"] = iter_vals[idxx, 1] / params["infectious_period"]
#     params["B_social"] = iter_vals[idxx, 0] * \
#         (params["N_social"] + params["N_const"])

#     ss, _ = find_ss(params)

#     ss.round(7)

#     I = ss[[2, 3]].sum()
#     B = ss[[1, 3, 5]].sum()

#     if np.isclose(I, 0):
#         if np.isclose(B, 0):
#             ss_categories.append(0)
#         else:
#             ss_categories.append(2)
#     else:
#         if np.isclose(B, 0):
#             ss_categories.append(1)
#         else:
#             ss_categories.append(3)

# ss_categories = np.array(ss_categories).reshape(grid_vals[0].shape)

# create_ss_plots_2(params, r0_b, r0_d, grid_vals, ss_categories, save=False)
