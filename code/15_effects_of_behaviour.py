#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:31:46 2023

@author: rya200
"""
# %%
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *

# %%


def calculate_endemic_i(params, R0_min=0.1, R0_max=5, step=0.1):
    R0_range = np.arange(R0_min, R0_max + step, step=step)

    params_no_behaviour = dict(params)
    params_no_behaviour["susc_B_efficacy"] = 0.0
    params_no_behaviour["inf_B_efficacy"] = 0.0

    params_full_behaviour = dict(params)
    params_full_behaviour["susc_B_efficacy"] = 1.0
    params_full_behaviour["inf_B_efficacy"] = 1.0

    I_no_behav = list()
    I_behav = list()

    for r0 in R0_range:
        beta = r0 * params["infectious_period"]

        params_no_behaviour["transmission"] = beta
        params_full_behaviour["transmission"] = beta

        ss_no_behav, _ = find_ss(params_no_behaviour)
        ss_behav, _ = find_ss(params_full_behaviour)

        I_no_behav.append(ss_no_behav[[2, 3]].sum())
        I_behav.append(ss_behav[[2, 3]].sum())

    I_no_behav = np.array(I_no_behav)
    I_behav = np.array(I_behav)

    return I_no_behav, I_behav, R0_range


def create_prevalence_plot(params, I_no_behav, I_behav, R0_range, save=False, dpi=600):

    # Assuming that a1=w2=0
    ss_tmp, _ = find_ss(params)
    B = ss_tmp[[1, 3, 5]].sum().round(2)

    text_min = I_behav[-1]
    text_max = I_no_behav[-1]

    I_diff = np.abs(I_no_behav - I_behav)[-1].round(3)

    plt.figure()
    plt.title(
        f"Endemic prevalence of infection\nfor endemic behaviour proportion = {B}")
    plt.plot(R0_range, I_no_behav, color="black",
             label="No behavioural protection")
    plt.plot(R0_range, I_behav, color="green",
             label="Full behavioural protection")
    plt.xlabel("$\\mathscr{R}_0^D$ ($\\beta/\\gamma$)")
    plt.ylabel("Endemic prevalence ($I$)")
    plt.legend()

    # plt.plot([5, 5], [text_min, text_max], linestyle=":", color="gray")
    # plt.text(x=5., y=text_min + (text_max-text_min)/2, s=f"{I_diff}")
    y_ticks = plt.yticks()[0]
    y_spacing = y_ticks[1] - y_ticks[0]

    plt.text(0., -y_spacing - 0.025,
             "Difference in prevalence when $\\mathscr{R}_0^D = 5$ is " + f"{I_diff}", va="top")

    if save:
        plt.savefig(f"../img/p_v_c_plots/endemic_i_prevalence_{int((B*100))}.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_prevalence_difference_plot(params, I_diff, R0_range, save=False, dpi=600, wrong=True):

    # Assuming that a1=w2=0
    ss_tmp, _ = find_ss(params)
    B = ss_tmp[[1, 3, 5]].sum().round(2)

    plt.figure()
    plt.title(
        f"Difference in peak prevalence\nfor endemic behaviour proportion = {B}")
    if not wrong:
        plt.plot(R0_range, I_diff, color="red")
        plt.xlabel("$\\mathscr{R}_0^D$ ($\\beta/\\gamma$)")
        plt.ylabel("Difference in peak prevalence($I$)")
    else:
        plt.plot(I_diff[:, 0], I_diff[:, 1], color="red")
        plt.xlabel("Prevalance when no behaviour")
        plt.ylabel("Prevalence when full behaviour")

    # plt.legend()

    if save:
        plt.savefig(f"../img/p_v_c_plots/difference_i_prevalence_{int((B*100))}.png",
                    dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_params(Bstar):
    model_params = dict()
    model_params["transmission"] = 1  # going to vary
    model_params["infectious_period"] = 1/1
    model_params["immune_period"] = 1/0.5
    model_params["av_lifespan"] = 0  # Turning off demography
    model_params["susc_B_efficacy"] = 0.
    model_params["inf_B_efficacy"] = 0.
    model_params["N_social"] = 0.
    model_params["B_social"] = 1/(1-Bstar)
    model_params["B_fear"] = 0.
    model_params["B_const"] = 0.
    model_params["N_const"] = 1.

    return model_params


def calculate_peak_prevelance_difference(params, R0_min=0.1, R0_max=5, step=0.1):
    R0_range = np.arange(R0_min, R0_max + step, step=step)

    params_no_behaviour = dict(params)
    params_no_behaviour["susc_B_efficacy"] = 0.0
    params_no_behaviour["inf_B_efficacy"] = 0.0

    params_full_behaviour = dict(params)
    params_full_behaviour["susc_B_efficacy"] = 1.0
    params_full_behaviour["inf_B_efficacy"] = 1.0

    IC = [1 - 2e-3, 1e-3, 1e-3, 0, 0, 0]
    t_start, t_end = [0, 100]

    I_diff = list()

    for r0 in R0_range:
        beta = r0 * params["infectious_period"]

        params_no_behaviour["transmission"] = beta
        params_full_behaviour["transmission"] = beta

        M_behaviour = bad(**params_full_behaviour)
        M_no_behaviour = bad(**params_no_behaviour)

        M_behaviour.run(IC=IC, t_start=t_start, t_end=t_end)
        M_no_behaviour.run(IC=IC, t_start=t_start, t_end=t_end)

        I_diff.append(M_no_behaviour.get_I().max() - M_behaviour.get_I().max())
        # I_diff.append([M_no_behaviour.get_I().max(),
        #               M_behaviour.get_I().max()])

    return np.array(I_diff), R0_range


# %%
B_min = 0
B_max = 0.60
B_step = 0.05
B_star_range = np.arange(B_min, B_max + B_step, step=B_step)


for b in B_star_range:
    model_params = create_params(b)

    I_no_behav, I_behav, R0_range = calculate_endemic_i(model_params)
    create_prevalence_plot(model_params, I_no_behav,
                           I_behav, R0_range, save=True)

# %%

for b in B_star_range:
    model_params = create_params(b)

    I_diff, R0_range = calculate_peak_prevelance_difference(
        model_params, step=0.05)
    create_prevalence_difference_plot(model_params, I_diff, R0_range,
                                      save=False, dpi=600, wrong=False)

# %%
model_params = create_params(0.2)
params_no_behaviour = dict(model_params)
params_no_behaviour["susc_B_efficacy"] = 0.0
params_no_behaviour["inf_B_efficacy"] = 0.0

params_full_behaviour = dict(model_params)
params_full_behaviour["susc_B_efficacy"] = 1.0
params_full_behaviour["inf_B_efficacy"] = 1.0

IC = [1 - 2e-3, 1e-3, 1e-3, 0, 0, 0]
t_start, t_end = [0, 100]

I_diff = list()

beta = 1.5 * model_params["infectious_period"]

params_no_behaviour["transmission"] = beta
params_full_behaviour["transmission"] = beta

M_behaviour = bad(**params_full_behaviour)
M_no_behaviour = bad(**params_no_behaviour)
M_behaviour.run(IC=IC, t_start=t_start, t_end=t_end)
M_no_behaviour.run(IC=IC, t_start=t_start, t_end=t_end)

plt.figure()
plt.plot(M_behaviour.get_I())
plt.plot(M_no_behaviour.get_I())
plt.show()
