#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:41:22 2023

@author: rya200
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:27:00 2023

@author: rya200
"""
# %% libraries

# %% parameter sweep and plotting functions




from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *
def parameter_sweep(parameter_code1, parameter_code2, sweep_params_input,
                    param_range_1=[0.1, 3.0],
                    param_range_2=[0.1, 8.0],
                    step1=0.1,
                    step2=0.1):
    sweep_params = dict(sweep_params_input)
    epi_r0_range = np.arange(
        param_range_1[0], param_range_1[1] + step1, step=step1)
    behav_r0_range = np.arange(
        param_range_2[0], param_range_2[1] + step2,  step=step2)

    xx, yy = np.meshgrid(epi_r0_range, behav_r0_range)

    r0_combos = np.array(np.meshgrid(epi_r0_range, behav_r0_range)).reshape(
        (2, len(epi_r0_range) * len(behav_r0_range))).T

    ss_list = list()
    R0_list = list()

    M3 = bad(**sweep_params)

    for idx in range(len(r0_combos)):
        b = r0_combos[idx, 0]
        w = r0_combos[idx, 1]

        sweep_params[parameter_code2] = w
        sweep_params[parameter_code1] = b

        ss, _ = find_ss(sweep_params)
        ss_list.append(ss)

        M3.update_params(**{parameter_code1: b, parameter_code2: w})
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

    return xx, yy, r0_combos, R0_list, BB, II


def plot_stead_state_parameter_sweeps(parameter_code1, parameter_code2,
                                      xx, yy, r0_combos,
                                      R0_list, BB, II, orig_param_dict,
                                      save=False, contour_img=True):
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
    lbl1 = code_to_label[parameter_code1]
    lbl2 = code_to_label[parameter_code2]

    save_lbl = "2_param_sweeps/ss_"
    for var in code_to_label.keys():
        if (var == parameter_code1) or (var == parameter_code2):
            continue
        save_lbl += code_to_label[var] + "_" + str(orig_param_dict[var]) + "_"

    # Behaviour steady states
    plt.figure()
    plt.title(f"Steady state of behaviour: {lbl1} and {lbl2} sweep")
    if contour_img:
        plt.contourf(xx, yy, np.array(BB).reshape(xx.shape),
                     # levels = lvls,
                     cmap=plt.cm.Blues)
        plt.colorbar(format=tkr.PercentFormatter(xmax=1, decimals=2))
    else:
        im = plt.imshow(np.array(BB).reshape(xx.shape),  cmap=plt.cm.Blues,
                        origin='lower',
                        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                        aspect="auto")
        ctr = plt.contour(xx, yy, np.array(BB).reshape(xx.shape),
                          colors="black",
                          alpha=0.5)
        cbar = plt.colorbar(
            im, format=tkr.PercentFormatter(xmax=1, decimals=2))
        cbar_lvls = ctr.levels[1:-1]
        cbar.add_lines(ctr)
        cbar.set_ticks(cbar_lvls)
    plt.xlabel(lbl1)
    plt.ylabel(lbl2)
    if save:
        plt.savefig("../img/" + save_lbl +
                    f"behaviour_{lbl1}_{lbl2}.png", dpi=600)
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.title(f"Steady state of Infection: {lbl1} and {lbl2} sweep")
    if contour_img:
        vec_II = np.array(II)
        full_rng = vec_II[vec_II.nonzero()[0]]
        full_rng = full_rng[np.isfinite(full_rng)]
        stp = full_rng.ptp()/6
        lvls = np.arange(full_rng.min(),
                         full_rng.max() + stp, step=stp)
        lvls = np.concatenate(([0], lvls))
        plt.contourf(xx, yy, vec_II.reshape(xx.shape),
                     levels=lvls, cmap=plt.cm.Reds)
        # plt.plot(new_R0, r0_combos[:, 1], "k:")
        plt.colorbar(format=tkr.PercentFormatter(xmax=1, decimals=2))
    else:
        vec_II = np.array(II)
        full_rng = vec_II[vec_II.nonzero()[0]]
        full_rng = full_rng[np.isfinite(full_rng)]
        stp = full_rng.ptp()/6
        lvls = np.arange(full_rng.min() + 5e-2,
                         full_rng.max() + stp, step=stp)
        # lvls = np.concatenate(([0], lvls))
        # plt.xlabel("Epidemic R0 (beta/gamma)")
        im = plt.imshow(vec_II.reshape(xx.shape),  cmap=plt.cm.Reds,
                        origin='lower',
                        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                        aspect="auto", vmin=0)
        ctr = plt.contour(xx, yy, vec_II.reshape(xx.shape),
                          levels=lvls,
                          colors="black", alpha=0.5)
        # plt.plot(new_R0, r0_combos[:, 1], "k:")
        cbar = plt.colorbar(
            im, format=tkr.PercentFormatter(xmax=1, decimals=2))
        cbar_lvls = ctr.levels[:-1]
        cbar.add_lines(ctr)
        cbar.set_ticks(cbar_lvls)
    plt.xlabel(lbl1)
    plt.ylabel(lbl2)
    if save:
        plt.savefig("../img/" + save_lbl +
                    f"infection_{lbl1}_{lbl2}.png", dpi=600)
        plt.close()
    else:
        plt.show()

    r0_lvls = np.arange(0, np.array(R0_list).max() + 1, step=1)
    plt.figure()
    plt.title(f"Behaviour affected R0: {lbl1} and {lbl2} sweep")
    plt.contourf(xx, yy, np.array(R0_list).reshape(
        xx.shape), levels=r0_lvls,  cmap=plt.cm.Greens)
    # plt.plot(new_R0, r0_combos[:, 1], "k:")
    plt.colorbar()
    # plt.xlabel("Epidemic R0 (beta/gamma)")
    plt.xlabel(lbl1)
    plt.ylabel(lbl2)
    if save:
        plt.savefig("../img/" + save_lbl +
                    f"R0_{lbl1}_{lbl2}.png", dpi=600)
        plt.close()
    else:
        plt.show()

# %%


heat_map_params = dict()
heat_map_params["transmission"] = 3
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
    # "transmission": ([0.1, 8], 0.1), # Essential done in 1 param sweeps
    "infectious_period": ([1, 14], 1),
    "immune_period": ([2 * 30, 8 * 30], 15),
    "susc_B_efficacy": ([0, 1], 0.05),
    "inf_B_efficacy": ([0, 1], 0.05),
    "N_social": (final_behav_R0_range, 0.1),
    "B_social": (final_behav_R0_range, 0.1),
    "B_fear": (final_behav_R0_range, 0.1),
    "B_const": (final_behav_R0_range, 0.1),
    "N_const": (final_behav_R0_range, 0.1)
}


# %% Create figures

save_figs = True

full_len = len(param_dict)
key_vals = list(param_dict.keys())
for i in range(full_len):
    kk1 = key_vals[i]
    for j in range(i, full_len):
        kk2 = key_vals[j]
        if kk1 == kk2:
            continue
        xx, yy, r0_combos, R0_list, BB, II = parameter_sweep(parameter_code1=kk1,
                                                             parameter_code2=kk2,
                                                             sweep_params_input=heat_map_params,
                                                             param_range_1=param_dict[kk1][0],
                                                             param_range_2=param_dict[kk2][0],
                                                             step1=param_dict[kk1][1],
                                                             step2=param_dict[kk2][1])
        if param_dict[kk1][1] > 0.2 or param_dict[kk2][1] > 0.2:
            ctr_img = True
        else:
            ctr_img = False
        plot_stead_state_parameter_sweeps(parameter_code1=kk1,
                                          parameter_code2=kk2,
                                          xx=xx, yy=yy, r0_combos=r0_combos,
                                          R0_list=R0_list,
                                          BB=BB, II=II,
                                          orig_param_dict=heat_map_params,
                                          save=save_figs,
                                          contour_img=ctr_img)

# %%
# kk1 = "transmission"
# kk2 = "immune_period"

# xx, yy, r0_combos, R0_list, BB, II = parameter_sweep(parameter_code1="transmission",
#                                                      parameter_code2="immune_period",
#                                                      sweep_params_input=heat_map_params,
#                                                      param_range_1=param_dict[kk1][0],
#                                                      param_range_2=param_dict[kk2][0],
#                                                      step1=param_dict[kk1][1],
#                                                      step2=param_dict[kk2][1])
# plot_stead_state_parameter_sweeps(parameter_code1=kk1,
#                                   parameter_code2=kk2,
#                                   xx=xx, yy=yy, r0_combos=r0_combos,
#                                   R0_list=R0_list,
#                                   BB=BB, II=II,
#                                   orig_param_dict=heat_map_params,
#                                   save=False)
# # %%

# plt.figure()
# plt.contourf(xx, yy, np.array(II).reshape(xx.shape), cmap=plt.cm.Blues)
# plt.show()

# plt.figure()
# plt.imshow(np.array(II).reshape(xx.shape), cmap=plt.cm.Blues)
# plt.show()
