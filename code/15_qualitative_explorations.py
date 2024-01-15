#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:47:25 2023

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

# %%


def make_plot_infections(model, IC, t_start, t_end, t_step, p=0, c=0, plottt="p"):
    MM.update_params(**{"inf_B_efficacy": p, "susc_B_efficacy": c})

    MM.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)

    if plottt == "p":
        col_val = p
        cm = plt.cm.Reds_r
    else:
        col_val = c
        cm = plt.cm.Blues_r

    if p == c and c == 0:
        lbl = "Blue hair"
    else:
        lbl = f"I: p = {MM.inf_B_efficacy}, c={MM.susc_B_efficacy}"

    col_val /= 2
    col_val -= 1e-5

    plt.plot(MM.t_range, MM.results[:, [2, 3]].sum(1),
             label=lbl,
             color=cm(col_val))


# %%

p = 0.75
c = 0.

model_params = dict()
model_params["transmission"] = 5/5
model_params["infectious_period"] = 5
model_params["immune_period"] = (8 * 30)
model_params["susc_B_efficacy"] = c
model_params["inf_B_efficacy"] = p
model_params["N_social"] = 0.5
model_params["B_social"] = 0.05*8
model_params["B_fear"] = 8
model_params["B_const"] = 0.01
model_params["N_const"] = 0.01

# model_params = dict()
# model_params["transmission"] = 1
# model_params["infectious_period"] = 5
# model_params["immune_period"] = (8 * 30)
# model_params["susc_B_efficacy"] = 0.
# model_params["inf_B_efficacy"] = 0.
# model_params["N_social"] = 0.5
# model_params["B_social"] = 0.6
# model_params["B_fear"] = 0.
# model_params["B_const"] = 0.0
# model_params["N_const"] = 0.1 * (0.6 - 0.5)

MM = bad(**model_params)


P = 1
Sb = 1e-3
Ib = 0
Rb = 0
In = 1e-3
Rn = 0
Sn = P - Sb - Ib - Rb - In - Rn

IC = [Sn, Sb, In, Ib, Rn, Rb]

t_start, t_end = [0, 300]
t_step = 1


plt.figure()
plt.title("I vs Rb")

tmp_dict = dict(model_params)
tmp_dict["susc_B_efficacy"] = c
tmp_dict["inf_B_efficacy"] = p
res = list()

R0b = np.arange(0, 3.1, step=0.1)

for rb in R0b:
    tmp_dict["B_social"] = rb * (tmp_dict["N_social"] + tmp_dict["N_const"])
    ss, _ = find_ss(tmp_dict)
    res.append(ss)
res = np.array(res)

plt.plot(R0b, res[:, 2], label=f"In, c = {c}, p = {p}", color="red")
plt.plot(R0b, res[:, 3], label=f"Ib, c = {c}, p = {p}", color="orange")


tmp_dict = dict(model_params)
tmp_dict["susc_B_efficacy"] = p
tmp_dict["inf_B_efficacy"] = c
res = list()


for rb in R0b:
    tmp_dict["B_social"] = rb * (tmp_dict["N_social"] + tmp_dict["N_const"])
    ss, _ = find_ss(tmp_dict)
    res.append(ss)
res = np.array(res)

plt.plot(R0b, res[:, 2], label=f"In, c = {p}, p = {c}",
         color="red", linestyle=":")
plt.plot(R0b, res[:, 3], label=f"Ib, c = {p}, p = {c}",
         color="orange", linestyle=":")


plt.legend()
plt.show()

plt.figure()
plt.title("I vs Rb pot 2")

tmp_dict = dict(model_params)
tmp_dict["susc_B_efficacy"] = c
tmp_dict["inf_B_efficacy"] = p
res = list()

R0b = np.arange(0, 3.1, step=0.1)

for rb in R0b:
    tmp_dict["B_social"] = rb * (tmp_dict["N_social"] + tmp_dict["N_const"])
    ss, _ = find_ss(tmp_dict)
    res.append(ss)
res = np.array(res)

plt.plot(R0b, res[:, 2] + res[:, 3], label=f"I, c = {c}, p = {p}", color="red")


tmp_dict = dict(model_params)
tmp_dict["susc_B_efficacy"] = p
tmp_dict["inf_B_efficacy"] = c
res = list()


for rb in R0b:
    tmp_dict["B_social"] = rb * (tmp_dict["N_social"] + tmp_dict["N_const"])
    ss, _ = find_ss(tmp_dict)
    res.append(ss)
res = np.array(res)

plt.plot(R0b, res[:, 2] + res[:, 3], label=f"I, c = {p}, p = {c}",
         color="red", linestyle=":")

plt.legend()
plt.show()

MM = bad(**model_params)
MM.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)
print(MM.Rzero())
plt.figure()
plt.title("Time series of infection for each strata")
plt.plot(MM.results[:, 2], label=f"In, c = {c}, p = {p}", color="red")
plt.plot(MM.results[:, 3], label=f"Ib, c = {c}, p = {p}", color="orange")
# plt.legend()
# plt.show()

MM1 = bad(**model_params)
MM1.update_params(**{"susc_B_efficacy": p, "inf_B_efficacy": c})
MM1.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)
print(MM1.Rzero())
# plt.figure()
plt.plot(MM1.results[:, 2], label=f"In, c = {p}, p = {c}",
         color="red",  linestyle=":")
plt.plot(MM1.results[:, 3], label=f"Ib, c = {p}, p = {c}",
         color="orange",  linestyle=":")
plt.legend()
plt.show()

plt.figure()
plt.title("Time series of infection")
plt.plot(MM.get_I(), label=f"I, p={p}, c = {c}")
plt.plot(MM1.get_I(), label=f"I, p={c}, c = {p}")
plt.legend()
plt.show()

# %%

pp = np.arange(0, 1.05, step=0.05)
cc = np.arange(0, 1.05, step=0.05)

mesh = np.meshgrid(pp, cc)

grid_vals = np.array(mesh).reshape(2, len(pp) * len(cc)).T

xx = mesh[0]
yy = mesh[1]


tmp_dict1 = dict(model_params)
tmp_dict2 = dict(model_params)
res = list()

for idx in range(len(grid_vals[:, 0])):
    tmp_dict1["susc_B_efficacy"] = grid_vals[idx, 0]
    tmp_dict2["inf_B_efficacy"] = grid_vals[idx, 0]
    tmp_dict1["inf_B_efficacy"] = grid_vals[idx, 1]
    tmp_dict2["susc_B_efficacy"] = grid_vals[idx, 1]
    M1 = bad(**tmp_dict1)
    M1.run(IC=IC, t_start=t_start, t_end=300, t_step=t_step)
    M2 = bad(**tmp_dict2)
    M2.run(IC=IC, t_start=t_start, t_end=300, t_step=t_step)

    res.append(np.abs(M1.get_I() - M2.get_I()).max())

# %%

plt.figure()
plt.imshow(np.array(res).reshape(xx.shape),
           origin='lower',
           extent=[xx.min(), xx.max(), yy.min(), yy.max()],
           aspect="auto")
plt.colorbar()
plt.show()


# for p in np.arange(0, 1.1, 0.1).round(1):
#     tmp_dict["susc_B_efficacy"] = p
#     tmp_dict["inf_B_efficacy"] = p
#     ss, _ = find_ss(tmp_dict)
#     res.append(ss)
# res = np.array(res)

# plt.figure()
# plt.plot(np.arange(0, 1.1, 0.1).round(1), res[:, [2, 3]].sum(1))
# plt.show()
# # MM.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)
# # MM.NGM()

# MM = bad(**model_params)
# MM.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)


# title = "B: " + str(MM.get_B()[-1].round(1)) + ", p = " + \
#     str(MM.inf_B_efficacy) + ", c = " + str(MM.susc_B_efficacy)

# plt.figure()
# plt.title("Infection, " + title)
# plt.plot(MM.get_I(), "red")
# plt.show()

# MM.update_params(**tmp_dict)
# MM.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)

# title = "B: " + str(MM.get_B()[-1].round(1)) + ", p = " + \
#     str(MM.inf_B_efficacy) + ", c = " + str(MM.susc_B_efficacy)

# plt.figure()
# plt.title("Infection, " + title)
# plt.plot(MM.get_I(), "red")
# plt.show()

# plt.figure()
# plt.title("R eff")
# plt.plot(MM.BA_Reff, "green")
# plt.plot([0, 1000], [1, 1], ":k")
# plt.show()

# plt.figure()
# plt.plot(MM.get_I(), MM.get_B())
# plt.show()

# plt.figure()
# plt.title("Infection progression, p vary")
# # plt.plot(MM.t_range, MM.results[:, [2, 3]].sum(1),
# #     label=f"I: p = {MM.inf_B_efficacy}, c={MM.susc_B_efficacy}",
# #     color = plt.cm.Reds_r(0))
# for pp in [0, 0.25, 0.5, 0.75, 1]:
#     make_plot_infections(MM, IC, t_start, t_end, t_step, p=pp, c=0.)
# plt.legend()
# plt.show()

# plt.figure()
# plt.title("Infection progression, c vary")
# # plt.plot(MM.t_range, MM.results[:, [2, 3]].sum(1),
# #     label=f"I: p = {MM.inf_B_efficacy}, c={MM.susc_B_efficacy}",
# #     color = plt.cm.Reds_r(0))
# for cc in [0, 0.25, 0.5, 0.75, 1]:
#     make_plot_infections(MM, IC, t_start, t_end,
#                          t_step, p=0., c=cc, plottt="c")
# plt.legend()
# plt.show()

# # %%
# plt.figure()
# plt.title("Infection rates")
# make_plot_infections(MM, IC, t_start, t_end, t_step, p=0, c=0)
# make_plot_infections(MM, IC, t_start, t_end, t_step, p=0.3, c=0.15)
# plt.legend()
# plt.show()

# p = 0.3
# c = 0.5
# plt.figure()
# plt.title("Infection rates")
# plt.plot(MM.get_I(), MM.get_B(),
#          # MM.t_range, MM.results[:, [2, 3]].sum(1),
#          label="Blue hair",
#          color="red", linestyle=":")
# print(
#     f"p = {MM.inf_B_efficacy}, c = {MM.susc_B_efficacy}, Istar={MM.get_I()[-1]}")
# MM.update_params(**{"inf_B_efficacy": p, "susc_B_efficacy": c})
# MM.run(IC=IC, t_start=t_start, t_end=t_end, t_step=t_step)
# plt.plot(MM.get_I(), MM.get_B(),
#          # MM.t_range, MM.results[:, [2, 3]].sum(1),
#          label=f"p={p}, c = {c}",
#          color="blue", linestyle="-")
# print(
#     f"p = {MM.inf_B_efficacy}, c = {MM.susc_B_efficacy}, Istar={MM.get_I()[-1]}")
# plt.legend()
# plt.show()
