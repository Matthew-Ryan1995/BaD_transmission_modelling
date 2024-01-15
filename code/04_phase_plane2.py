#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:42:09 2023

@author: rya200
"""


# %% Packages/libraries
import numpy as np
import scipy as spi
# import matplotlib.pyplot as plt
import os
import pandas as pd
from msir_4 import msir
import plotnine as gg
import time

working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %% Set up

tic = time.time()

TS = 1.0
ND = 200.0

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)

# Inital conditions
N = 1e6
# Note the order of conditions (M-N)
S0_m = 0.
I0_m = 0
I0_n = 1  # 1% start infected
R0_n = 0
R0_m = 0
S0_n = N - S0_m - I0_m - I0_n - R0_n - R0_m
FS = 0
init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n, FS)

# w1 = 8

# # Enter custom params
# cust_params = dict()
# cust_params["transmission"] = 3.2
# cust_params["infectious_period"] = 1/0.4
# # cust_params["immune_period"] = 0
# cust_params["av_lifespan"] = 0
# cust_params["susc_mask_efficacy"] = 0.85
# cust_params["inf_mask_efficacy"] = 0.85
# cust_params["nomask_social"] = 0.5
# cust_params["nomask_fear"] = 0
# cust_params["mask_social"] = 0.05 * w1
# cust_params["mask_fear"] = w1
# model = msir(**cust_params)

# # Run integrator, convert results to long format
# RES = spi.integrate.solve_ivp(fun=model.run,
#                               t_span=[t_start, t_end],
#                               y0=init_cond,
#                               t_eval=t_range)
# dat = RES.y.T

# Rt = list(map(lambda t: model.NGM(dat[t, :]), range(len(t_range))))

# switch_time = next(i for i, V in enumerate(Rt) if V <= 1)
# tt = t_range[switch_time]
# %%d

# plt.plot(dat[:, 2] + dat[:, 3])
# tmp = spi.signal.find_peaks(dat[:, 2] + dat[:, 3], height=(1, None))
# print("Number of peaks is %d" % (tmp[0].size))

# %%

# c = p = np.array([0.0, 0.7, 0.85, 1.0])
c = p = np.arange(start=0, stop=1.1, step=0.1)
# c = p = np.array([0.85])

R0 = w2 = np.arange(0, 10, step=0.5)
R0 = R0[R0 > 1]  # No peaks at all for R0 = 0 or 1

beta = 0.4 * R0


params = np.array(np.meshgrid(c, p, beta, w2)).reshape(
    4, len(c) * len(p) * len(w2) * len(beta)).T

params = pd.DataFrame(params, columns=["c", "p", "beta", "w2"])
params = params.round(1)

params.reset_index()


def calculate_peaks(idx_row):
    idx, row = idx_row

    new_params = dict()

    new_params["transmission"] = row["beta"]
    new_params["susc_mask_efficacy"] = row["c"]
    new_params["inf_mask_efficacy"] = row["p"]
    new_params["mask_social"] = 0.05 * row["w2"]
    new_params["mask_fear"] = row["w2"]
    new_params["infectious_period"] = 1/0.4
    # cust_params["immune_period"] = 0
    new_params["av_lifespan"] = 0
    new_params["nomask_social"] = 0.5
    new_params["nomask_fear"] = 0.00 * 0.5

    model = msir(**new_params)

    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=init_cond,
                                  t_eval=t_range)
    dat = RES.y.T

    return dat[-1, 6]


# %%
peaks = list(map(calculate_peaks, params.iterrows()))

params["peaks"] = peaks
params["peaks"] = params.peaks/N
# params["peaks"] = pd.Categorical(params.peaks)

params["R0"] = params.beta/0.4

toc = time.time()

print("Script time is %f" % (toc - tic))

# %%


pp = (gg.ggplot(data=params[params.peaks >= 0], mapping=gg.aes(x="R0", y="w2", fill="peaks")) +
      gg.geom_tile() +
      gg.labs(x="Basic reproduction number", y="Fear of disease", fill="Final Size") +
      # gg.scale_fill_distiller(palette="RdBu", direction=1, limits=[0, 1]) +
      gg.scale_fill_continuous(limts=[0, 1]) +
      gg.theme_classic() +
      gg.facet_grid("c ~ p", labeller=gg.label_both))

print(pp)

pp.save(filename="phase_plot_final_size.png", width=15, height=15)

params.to_csv("phase_plot_final_size_csv.csv")

# %%
