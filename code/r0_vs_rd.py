#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a bad idea.  Only straight line. R0D vs R0

how about R0B vs R0

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
# m_params["B_const"] = 0.5  # 0.7
# m_params["N_const"] = 0.9  # 0.9


# %%
r0_d = np.arange(0, 5, step=0.1)

r0 = list()
pars = dict(m_params)
M = bad(**pars)

mult_var = M.Rzero()

r0 = r0_d * mult_var

# plt.figure()
# plt.plot(r0_d, r0)
# plt.plot([0, 5], [0, 5])
# plt.xlabel("$\\mathscr{R}_0^D$")
# plt.ylabel("$\\mathscr{R}_0$")
# plt.show()


# %%
R0_d = 10

pars = dict(m_params)
pars["transmission"] = R0_d/pars["infectious_period"]

M = bad(**pars)

R0_b = np.arange(0, 5, step=0.1)
R0 = list()

for rr in R0_b:
    pars["B_social"] = rr * \
        (pars["N_const"] + pars["B_social"])
    M.update_params(**pars)
    R0.append(M.Rzero())

R0 = np.array(R0)

# plt.figure()
# plt.plot(R0_b, R0)
# plt.xlabel("$\\mathscr{R}_0^B$")
# plt.ylabel("$\\mathscr{R}_0$")
# plt.show()

# %%


def find_i_diff(params):

    params_no_behaviour = dict(params)
    params_no_behaviour["susc_B_efficacy"] = 0.0
    params_no_behaviour["inf_B_efficacy"] = 0.0

    params_full_behaviour = dict(params)
    params_full_behaviour["susc_B_efficacy"] = 1.0
    params_full_behaviour["inf_B_efficacy"] = 1.0

    ss_no_behav, _ = find_ss(params_no_behaviour)
    ss_behav, _ = find_ss(params_full_behaviour)

    I_diff = ss_no_behav[[2, 3]].sum() - ss_behav[[2, 3]].sum()
    return I_diff


def create_params(Bstar, R0=5):
    model_params = load_param_defaults()
    model_params["transmission"] = R0
    # model_params["infectious_period"] = 1/1
    # model_params["immune_period"] = 1/0.4
    # model_params["av_lifespan"] = 0  # Turning off demography
    model_params["susc_B_efficacy"] = 0.
    model_params["inf_B_efficacy"] = 0.
    model_params["N_social"] = 0.
    model_params["B_fear"] = 0.
    model_params["B_const"] = 0.
    # model_params["transmission"] = R0
    # model_params["infectious_period"] = 1/1
    # model_params["immune_period"] = 1/0.5
    # model_params["av_lifespan"] = 0  # Turning off demography
    # model_params["susc_B_efficacy"] = 0.
    # model_params["inf_B_efficacy"] = 0.
    # model_params["N_social"] = 0.
    # model_params["B_fear"] = 0.
    # model_params["B_const"] = 0.

    if Bstar == 1:
        model_params["N_const"] = 0
        model_params["B_social"] = 1
    else:
        model_params["B_social"] = 1/(1-Bstar)
        model_params["N_const"] = 1.

    return model_params


def prevalence_change_plot(disease_type, target_reduction=20, save=False):
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

    B_min = 0.0
    B_max = 1.0
    B_step = 0.01
    B_star_range = np.arange(B_min, B_max + B_step, step=B_step)
    I_diff = list()

    for b in B_star_range:
        model_params = create_params(b, R0=R0)

        I_diff.append(find_i_diff(model_params))

    I_diff = np.array(I_diff)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Endemic behaviour prevalence (B*)')
    ax1.set_ylabel(
        'Absolute difference in \nendemic disease prevalence', color=color)
    ax1.plot(B_star_range, I_diff, color="black", label="covid-like")
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xlim(0, 1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    prop_change = I_diff/I_diff.max() * 100
    idx_change = next(i for i in range(len(prop_change))
                      if prop_change[i] > target_reduction)
    B_star_target = B_star_range[idx_change]

    idx_change_100 = next(i for i in range(len(prop_change))
                          if prop_change[i] >= 99)  # Numerical accuracy
    B_star_100 = B_star_range[idx_change_100]

    print(
        f"$B^*$ for {target_reduction} reduction is {np.round(B_star_target * 100, 2)}")
    print(f"$B^*$ for 100 reduction is {np.round(B_star_100 * 100, 2)}")

    text_label = "$B^* =$" + f"{np.round(B_star_target * 100, 1)}"

    ax2.set_ylabel('Percentage change in \nendemic disease prevalence',
                   color=color, rotation=-90, va="center")
    ax2.plot(B_star_range, I_diff/I_diff.max() * 100, color="black")
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.plot([0, 1], [20, 20], ":k")
    # ax2.plot([B_star_target, B_star_target], [0, 100], ":k")
    # ax2.plot([B_star_100, B_star_100], [0, 100], ":k")
    # ax2.plot([B_star_target, B_star_target], [target_reduction, target_reduction],
    #          marker="o", markersize=10, markerfacecolor="none", markeredgecolor="red")
    # ax2.text(text_factor*B_star_target, target_reduction, text_label,
    #          horizontalalignment='left', verticalalignment='center')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(title)
    if save:
        plt.savefig(
            f"../img/endemic_difference/difference_plot_{disease_type}.png",
            bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


prevalence_change_plot("covid_like", save=False)
prevalence_change_plot("flu_like", save=False)
# plt.figure()
# plt.plot(B_star_range, I_diff)
# plt.xlabel("$B^*$")
# plt.ylabel("$I^*$ diff")
# plt.show()


# %%

# nu = 0.5
# g = 1
# beta = 10

# print(nu*(beta-g)/(beta*(g+nu)))

# # %%


# plot_params_baseline = dict()
# plot_params_baseline["transmission"] = 5
# plot_params_baseline["infectious_period"] = 1/1
# plot_params_baseline["immune_period"] = 1/0.5
# plot_params_baseline["av_lifespan"] = 0  # Turning off demography
# plot_params_baseline["susc_B_efficacy"] = 0.5
# plot_params_baseline["inf_B_efficacy"] = 0.3
# plot_params_baseline["N_social"] = 0.9
# plot_params_baseline["B_fear"] = 0.
# plot_params_baseline["B_const"] = 0.
# plot_params_baseline["N_const"] = 0.5

# plot_params_baseline["B_social"] = 1.9 * \
#     (plot_params_baseline["N_social"] + plot_params_baseline["N_const"])

# M = bad(**plot_params_baseline)

# IC = [1-2e-4, 1e-4, 1e-4, 0, 0, 0]
# # IC = [1-1e-4, 0, 1e-4, 0, 0, 0]

# M.run(IC, 0, 250)

# plt.figure()
# plt.plot(M.get_I())
# plt.show()

# plt.figure()
# plt.plot(M.get_B())
# plt.show()

# # %%

# N = M.results[:, [0, 2, 4]].sum(1)
# I = M.results[:, [2, 3]].sum(1)

# x = M.rate_to_mask(1-N, I) * N
# y = M.rate_to_no_mask(N, 1-I) * (1-N)

# plt.figure()
# plt.plot(x, y)
# plt.plot([x.min(), x.max()], [x.min(), x.max()])
# plt.xlabel("$\\omega$")
# plt.ylabel("$\\alpha$")
# plt.show()
