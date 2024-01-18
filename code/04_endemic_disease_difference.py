#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Look at the differences in endemic disease prevalence between a population with no behaviour and one with a 
fully protective behaviour

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

# %% Define parameters

m_params = load_param_defaults()

save_now = True


# %%

# Calculate the difference in disease prevalence
def find_i_diff(params):

    beta = params["transmission"]
    gamma = 1/params["infectious_period"]
    nu = 1/params["immune_period"]

    n_behav_I = nu*(beta-gamma)/(beta*(gamma+nu))

    params_full_behaviour = dict(params)
    params_full_behaviour["susc_B_efficacy"] = 1.0
    params_full_behaviour["inf_B_efficacy"] = 1.0

    ss_behav, _ = find_ss(params_full_behaviour)

    I_diff = n_behav_I - ss_behav[[2, 3]].sum()
    return I_diff


# Create a parameters set
def create_params(w, R0=5):
    model_params = load_param_defaults()
    model_params["transmission"] = R0
    model_params["B_social"] = w
    model_params["susc_B_efficacy"] = 1.0
    model_params["inf_B_efficacy"] = 1.0

    return model_params

# Plot the differences


def prevalence_change_plot(disease_type, target_reduction=20, save=False):
    if disease_type == "covid_like":
        R0 = 8.2
        title = "Covid-like illness ($\\mathscr{R}_0^D = 8.2$)"
        text_factor = 1.2
    elif disease_type == "flu_like":
        R0 = 1.5
        title = "Influenza-like illness ($\\mathscr{R}_0^D = 1.5$)"
        text_factor = 2
    else:
        R0 = disease_type
        title = "$\\mathscr{R}_0^D =$"+f"{R0}"

    init_param = create_params(w=0.4, R0=R0)
    ss, _ = find_ss(init_param)
    B_vert = ss[[1, 3, 5]].sum()
    print(f"Bvert is {B_vert.round(2)}")

    B_star_range = [0]
    ww = np.arange(0, 10, step=0.1)
    I_diff = [0]

    for w in ww:
        model_params = create_params(w=w, R0=R0)
        tmp, _ = find_ss(model_params)
        B_star_range.append(tmp[[1, 3, 5]].sum())

        I_diff.append(find_i_diff(model_params))
    B_star_range.append(1.0)
    B_star_range = np.array(B_star_range)
    I_diff.append(max(I_diff))
    I_diff = np.array(I_diff)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Endemic behaviour prevalence (B*)')
    ax1.set_ylabel(
        'Absolute difference in \nendemic disease prevalence', color=color)
    ax1.plot(B_star_range, I_diff, color="black", label="covid-like")
    # ax1.scatter(B_star_range, I_diff, color="red")
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
    ax2.plot([B_vert, B_vert], [0, 100], ":k")
    # ax2.plot([0, 1], [20, 20], ":k")
    # ax2.plot([B_star_target, B_star_target], [0, 100], ":k")
    # ax2.plot([B_star_100, B_star_100], [0, 100], ":k")
    # ax2.plot([B_star_target, B_star_target], [target_reduction, target_reduction],
    #          marker="o", markersize=10, markerfacecolor="none", markeredgecolor="red")
    # ax2.text(text_factor*B_star_target, target_reduction, text_label,
    #          horizontalalignment='left', verticalalignment='center')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title(title)
    if save:
        plt.savefig(
            f"../img/endemic_difference/difference_plot_{disease_type}.png",
            bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


prevalence_change_plot("covid_like", save=save_now)
prevalence_change_plot("flu_like", save=save_now)
