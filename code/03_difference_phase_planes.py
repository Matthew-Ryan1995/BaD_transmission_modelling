#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 07:09:58 2023

Phase planes to investigate infection trajectory to copare no behaviour with full behaviour.

@author: Matt Ryan
"""

from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
from BaD import *
# %% parameters

save_file = True

# %% Functions


def create_params(R0=5, p=1, c=1):
    model_params = load_param_defaults()
    model_params["transmission"] = R0/model_params["infectious_period"]
    model_params["susc_B_efficacy"] = c
    model_params["inf_B_efficacy"] = p

    return model_params


# Add arrows
# From https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot

def add_arrow(line, position=None, direction='right', size=15, color=None, linestyle=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()
    if linestyle is None:
        linestyle = line.get_linestyle()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->",
                                       color=color, lw=3),
                       size=size
                       )


def generate_phase_plane(disease_type, position_b=0.95, position_n=0.95, save=False):
    if disease_type == "covid_like":
        R0 = 8.2
        title = "Covid-like illness ($\\mathscr{R}_0^D = 8.2$)"
    elif disease_type == "flu_like":
        R0 = 1.5
        title = "Influenza-like illness ($\\mathscr{R}_0^D = 1.5$)"
    else:
        R0 = disease_type
        title = "$\\mathscr{R}_0^D =$"+f"{R0}"

    m_params = create_params(R0=R0)

    Sb = 1e-3
    In = 1e-3
    Ib = Rb = 0
    Rn = [0.0]

    t_start, t_end = [0, 900]

    res_i = list()
    res_s = list()
    res_b = list()
    res_i_n = list()
    res_s_n = list()
    res_b_n = list()
    M = bad(**m_params)
    M2 = bad(**m_params)
    M2.update_params(**{"inf_B_efficacy": 0.0, "susc_B_efficacy": 0.0})
    for rn in Rn:
        Sn = 1-Sb-In-Ib-Rb-rn
        IC = [Sn, Sb, In, Ib, rn, Rb]

        M.run(IC, t_start, t_end, t_step=0.1)
        M2.run(IC, t_start, t_end, t_step=0.1)

        res_i.append(M.get_I())
        res_s.append(M.get_S())
        res_b.append(M.get_B())
        res_i_n.append(M2.get_I())
        res_s_n.append(M2.get_S())
        res_b_n.append(M2.get_B())

    plt.figure()
    # plt.title(title)
    for idx, ii in enumerate(res_i):
        ss = res_s[idx]
        ii_n = res_i_n[idx]
        ss_n = res_s_n[idx]
        l1 = plt.plot(ss, ii, label="Fully protective behaviour",
                      color="blue")[0]
        add_arrow(l1, color="blue", position=position_b)
        l2 = plt.plot(ss_n, ii_n, linestyle=":", color="orangered",
                      label="No protective behaviour")[0]
        add_arrow(l2, color="orangered", position=position_n,
                  linestyle=":", size=20)
    plt.xlabel('Proportion of susceptibles (S)')
    plt.ylabel('Proportion of infectious (I)')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    if save:
        plt.savefig(
            f"../img/endemic_difference/phase_plane_{disease_type}.png",
            bbox_inches="tight", dpi=600)
        plt.close()
    else:
        plt.show()


generate_phase_plane("covid_like", position_b=0.8, save=save_file)
generate_phase_plane("flu_like", position_b=0.95, save=save_file)
