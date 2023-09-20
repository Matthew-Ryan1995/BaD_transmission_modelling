#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:27:00 2023

@author: rya200
"""
# %%
import scipy.ndimage
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

# %% parameter sweep and plotting functions


def parameter_sweep(parameter_code, sweep_params_input,  epi_r0=False,
                    epi_range=[0.1, 8.0], behav_range=[0.1, 3.0],
                    epi_stp=0.1, behav_step=0.1):
    sweep_params = dict(sweep_params_input)
    epi_r0_range = np.arange(
        epi_range[0], epi_range[1] + epi_stp, step=epi_stp)
    behav_r0_range = np.arange(
        behav_range[0], behav_range[1] + behav_step,  step=behav_step)

    xx, yy = np.meshgrid(epi_r0_range, behav_r0_range)

    r0_combos = np.array(np.meshgrid(epi_r0_range, behav_r0_range)).reshape(
        (2, len(epi_r0_range) * len(behav_r0_range))).T

    ss_list = list()
    R0_list = list()

    M3 = bad(**sweep_params)

    for idx in range(len(r0_combos)):
        b = r0_combos[idx, 0]
        w = r0_combos[idx, 1]
        ww = w

        if epi_r0:
            bb = b * (1/sweep_params["infectious_period"])
        else:
            M3.update_params(**{parameter_code: ww})
            inter_r0 = M3.Rzero()
            multi_val2 = inter_r0/M3.transmission
            bb = b * (1/multi_val2)

        sweep_params[parameter_code] = ww
        sweep_params["transmission"] = bb

        ss, _ = find_ss(sweep_params)
        ss_list.append(ss)

        M3.update_params(**{"transmission": bb, parameter_code: ww})
        R0_list.append(M3.Rzero())
        # print(sweep_params)

    def calc_B(X):
        xx = X[1]
        B = xx[[1, 3, 5]].sum()
        return B

    def calc_I(X):
        xx = X[1]
        B = xx[[2, 3]].sum()
        return B

    BB = list(map(calc_B, enumerate(ss_list)))
    II = list(map(calc_I, enumerate(ss_list)))

    if epi_r0:
        beta_vals = list()
        for idx in range(len(r0_combos)):
            b = r0_combos[idx, 0]
            w = r0_combos[idx, 1]
            bb = b * (1/sweep_params["infectious_period"])
            ww = w

            M3.update_params(**{"transmission": bb, parameter_code: ww})

            tmp = M3.Rzero()/bb

            beta_vals.append(1/tmp)

        new_R0 = np.array(beta_vals) * sweep_params["infectious_period"]
    else:
        new_R0 = np.nan

    return xx, yy, r0_combos, R0_list, BB, II, new_R0


def plot_stead_state_parameter_sweeps(parameter_code, xx, yy, r0_combos,
                                      R0_list, BB, II, new_R0, orig_param_dict,
                                      epi_r0=False, save=False):
    code_to_label = {
        "transmission": "beta",
        "infectious_period": "gamma_inv",
        "immune_period": "nu_inv",
        "susc_B_efficacy": "c",
        "inf_B_efficacy": "p",
        "N_social": "a1",
        "B_social": "w1",
        "B_fear": "w2",
        "B_const": "w3",
        "N_const": "a3"
    }
    lbl = code_to_label[parameter_code]

    save_lbl = f"1_param_sweeps/epiR0_{epi_r0}_ss_"
    for var in code_to_label.keys():
        if (var == parameter_code) or (var == "transmission"):
            continue
        save_lbl += code_to_label[var] + "_" + str(orig_param_dict[var]) + "_"

    # Behaviour steady states
    plt.figure()
    plt.title(f"Steady state of behaviour: {lbl} sweep")
    # plt.contourf(xx, yy, np.array(BB).reshape(xx.shape),
    #              # levels = lvls,
    #              cmap=plt.cm.Blues)
    im = plt.imshow(np.array(BB).reshape(xx.shape),  cmap=plt.cm.Blues,
                    origin='lower',
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    aspect="auto")
    ctr = plt.contour(xx, yy, np.array(BB).reshape(xx.shape),
                      # levels = lvls,
                      # cmap=plt.cm.Blues,
                      colors="black",
                      alpha=0.5)
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = ctr.levels[1:-1]
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    if epi_r0:
        plt.xlabel("$\\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
        plt.plot(new_R0, r0_combos[:, 1], "k:")
    else:
        plt.xlabel("Behaviour affected R0")
    plt.ylabel(lbl)
    if save:
        plt.savefig("../img/" + save_lbl +
                    f"behaviour_{lbl}.png", dpi=600)
        plt.close()
    else:
        plt.show()

    vec_II = np.array(II)
    stp = vec_II[vec_II.nonzero()[0]].ptp()/6
    lvls = np.arange(vec_II[vec_II.nonzero()[0]].min() + 5e-2,
                     vec_II.max() + stp, step=stp)
    # lvls = np.concatenate(([0], lvls))

    plt.figure()
    plt.title(f"Steady state of Infection: {lbl} sweep")
    # plt.contourf(xx, yy, vec_II.reshape(xx.shape),
    #              levels=lvls, cmap=plt.cm.Reds)
    im = plt.imshow(vec_II.reshape(xx.shape),  cmap=plt.cm.Reds,
                    origin='lower',
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    aspect="auto", vmin=0)
    ctr = plt.contour(xx, yy, vec_II.reshape(xx.shape),
                      levels=lvls,
                      colors="black", alpha=0.5)
    # plt.plot(new_R0, r0_combos[:, 1], "k:")
    cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
    cbar_lvls = ctr.levels[:-1]
    cbar.add_lines(ctr)
    cbar.set_ticks(cbar_lvls)
    # plt.xlabel("Epidemic R0 (beta/gamma)")
    if epi_r0:
        plt.xlabel("$\\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
        plt.plot(new_R0, r0_combos[:, 1], "k:")
    else:
        plt.xlabel("Behaviour affected R0")
    plt.ylabel(lbl)
    if save:
        plt.savefig("../img/" + save_lbl +
                    f"infection_{lbl}.png", dpi=600)
        plt.close()
    else:
        plt.show()

    if epi_r0:
        r0_lvls = np.arange(0, np.array(R0_list).max() + 1, step=1)
        plt.figure()
        plt.title(f"Behaviour affected R0: {lbl} sweep")
        plt.contourf(xx, yy, np.array(R0_list).reshape(
            xx.shape), levels=r0_lvls,  cmap=plt.cm.Greens)
        # plt.imshow(np.array(R0_list).reshape(xx.shape),
        #            cmap=plt.cm.Greens,
        #            origin='lower',
        #            extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        #            aspect="auto")
        # plt.plot(new_R0, r0_combos[:, 1], "k:")
        plt.colorbar()
        # plt.xlabel("Epidemic R0 (beta/gamma)")
        if epi_r0:
            plt.xlabel("$\\mathscr{R}_0^{D}$ ($\\beta/\\gamma$)")
            plt.plot(new_R0, r0_combos[:, 1], "k:")
        else:
            plt.xlabel("Behaviour affected R0")
        plt.ylabel(lbl)
        if save:
            plt.savefig("../img/" + save_lbl +
                        f"R0_{lbl}.png", dpi=600)
            plt.close()
        else:
            plt.show()

# %%


heat_map_params = dict()
heat_map_params["transmission"] = 1
heat_map_params["infectious_period"] = 1/1
heat_map_params["immune_period"] = 1/0.5
heat_map_params["av_lifespan"] = 0  # Turning off demography
heat_map_params["susc_B_efficacy"] = 0.5
heat_map_params["inf_B_efficacy"] = 0.3
heat_map_params["N_social"] = 0.2
heat_map_params["B_social"] = 1.3
heat_map_params["B_fear"] = 0.5
heat_map_params["B_const"] = 0.7
heat_map_params["N_const"] = 0.9

# Different set of params
# heat_map_params = dict()
# heat_map_params["transmission"] = 1
# heat_map_params["infectious_period"] = 1/0.4
# heat_map_params["immune_period"] = 1/(8*30)
# heat_map_params["av_lifespan"] = 0  # Turning off demography
# heat_map_params["susc_B_efficacy"] = 0.4
# heat_map_params["inf_B_efficacy"] = 0.8
# heat_map_params["N_social"] = 0.5
# heat_map_params["B_social"] = 0.05 * 8
# heat_map_params["B_fear"] = 8
# heat_map_params["B_const"] = 0.01
# heat_map_params["N_const"] = 0.01


R0_minmax = [0.1, 8]

final_behav_R0_range = [0.1, 8]

param_dict = {
    # "transmission": ([0.1, 8], 0.1),
    # "infectious_period": ([2, 14], 1),
    # "immune_period": ([2 * 30, 8 * 30], 15),
    "susc_B_efficacy": ([0, 1], 0.05),
    "inf_B_efficacy": ([0, 1], 0.05),
    "N_social": (final_behav_R0_range, 0.1),
    "B_social": (final_behav_R0_range, 0.1),
    "B_fear": (final_behav_R0_range, 0.1),
    "B_const": (final_behav_R0_range, 0.1),
    "N_const": (final_behav_R0_range, 0.1)
}


# %% Create figures

epi_r0 = True  # Plot beta/gamma on x axis or not
save_figs = True

for kk in param_dict.keys():

    xx, yy, r0_combos, R0_list, BB, II, new_R0 = parameter_sweep(kk, heat_map_params,
                                                                 epi_r0=epi_r0,
                                                                 epi_range=R0_minmax,
                                                                 behav_range=param_dict[kk][0],
                                                                 behav_step=param_dict[kk][1])
    plot_stead_state_parameter_sweeps(kk, xx, yy, r0_combos, R0_list,
                                      BB, II, new_R0, orig_param_dict=heat_map_params,
                                      epi_r0=epi_r0, save=save_figs)


# # %%
# vec_II = np.array(II)
# stp = vec_II[vec_II.nonzero()[0]].ptp()/6
# lvls = np.arange(vec_II[vec_II.nonzero()[0]].min() + 1e-2,
#                  vec_II.max() + stp, step=stp)
# # lvls = np.concatenate(([0], lvls))

# # tmp = vec_II.reshape(xx.shape)
# # tmp = scipy.ndimage.zoom(tmp, 3)
# # tmpx = scipy.ndimage.zoom(xx, 3)
# # tmpy = scipy.ndimage.zoom(yy, 3)

# plt.figure()
# # plt.contourf(xx, yy, vec_II.reshape(xx.shape),
# #              levels=lvls, cmap=plt.cm.Reds)

# ctr = plt.contour(xx, yy, vec_II.reshape(xx.shape),
#                    levels=lvls,
#                   colors="black", alpha=0.5)
# im = plt.imshow(tmp,  cmap=plt.cm.Reds,
#                 origin='lower',
#                 extent=[xx.min(), xx.max(), yy.min(), yy.max()],
#                 aspect="auto", vmin=0)
# # plt.plot(new_R0, r0_combos[:, 1], "k:")
# cbar = plt.colorbar(im, format=tkr.PercentFormatter(xmax=1, decimals=2))
# cbar_lvls = ctr.levels[1:-1]
# cbar.add_lines(ctr)
# cbar.set_ticks(cbar_lvls)
# plt.show()
