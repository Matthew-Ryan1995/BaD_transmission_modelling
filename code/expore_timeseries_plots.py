#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:31:49 2023

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

pars = load_param_defaults()
pars["transmission"] = 3
pars["B_fear"] = 0

In = Sb = 1e-3
Rn = Rb = Ib = 0
Sn = 1 - Ib - Sb

PP = np.array([Sn, Sb, In, Ib, Rn, Rb])

t_start, t_end = [0, 50]

M = bad(**pars)

M.run(IC=PP, t_start=t_start, t_end=t_end,  t_step=0.1)

# %%

# plt.figure()
# plt.plot(M.t_range, M.get_I())
# plt.show()
# plt.figure()
# plt.plot(M.t_range, M.get_B())
# plt.show()


fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (t)')
ax1.set_ylabel(
    'Disease prevalence', color=color)
ax1.plot(M.t_range, M.get_I(), color=color, label="covid-like")
# ax1.scatter(B_star_range, I_diff, color="red")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
# we already handled the x-label with ax1

ax2.set_ylabel('Behaviour prevalence',
               color=color, rotation=-90, va="center")
ax2.plot(M.t_range, M.get_B(), color=color, linestyle=":")
ax2.tick_params(axis='y', labelcolor=color)
# ax2.plot([0, 1], [20, 20], ":k")
# ax2.plot([B_star_target, B_star_target], [0, 100], ":k")
# ax2.plot([B_star_100, B_star_100], [0, 100], ":k")
# ax2.plot([B_star_target, B_star_target], [target_reduction, target_reduction],
#          marker="o", markersize=10, markerfacecolor="none", markeredgecolor="red")
# ax2.text(text_factor*B_star_target, target_reduction, text_label,
#          horizontalalignment='left', verticalalignment='center')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# plt.figure()
# plt.plot(M.get_I(), M.get_B())
# plt.show()
