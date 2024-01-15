#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:08:25 2023

Calculate the peaks from an epidemic for different values of the social influence and fear for masks.

fixing:
    - gamma = 0.4
    - a1 = 0.5
    - a2 = 0
    - beta/gamma = 7

Do this in parallel?

@author: rya200
"""
# %% Packages
from multiprocessing import pool
from multiprocessing import cpu_count
import numpy as np
import scipy as spi
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

# %%

# c = p = np.array([0.0, 0.7, 0.85, 1.0])
c = p = np.arange(start=0, stop=1.1, step=0.1)
# c = p = np.array([0.85])

w1 = np.arange(0, 0.5, step=0.025)
w2 = np.arange(0, 10, step=0.5)
# w1 = 0.05 * w2


params = np.array(np.meshgrid(c, p, w1, w2)).reshape(
    4, len(c) * len(p) * len(w2) * len(w1)).T

params = pd.DataFrame(params, columns=["c", "p", "w1", "w2"])
params = params.round(3)

params.reset_index()
# params.w1 = 0.05 * params.w1


def event(t, y):
    if t > 10:
        ans = y[2] + y[3] - 1
    else:
        ans = 1
    return ans


event.terminal = True


def calculate_peaks(idx_row):
    idx, row = idx_row

    new_params = dict()

    new_params["transmission"] = 7 * 0.4
    new_params["susc_mask_efficacy"] = row["c"]
    new_params["inf_mask_efficacy"] = row["p"]
    new_params["mask_social"] = row["w1"]
    new_params["mask_fear"] = row["w2"]
    new_params["infectious_period"] = 1/0.4
    # cust_params["immune_period"] = 0
    new_params["av_lifespan"] = 0
    new_params["nomask_social"] = 0.5
    new_params["nomask_fear"] = 0.0

    model = msir(**new_params)

    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=init_cond,
                                  t_eval=t_range,
                                  events=[event])
    dat = RES.y.T

    peak_count = spi.signal.find_peaks(
        dat[:, 2] + dat[:, 3], height=(1, None), prominence=(100, None))

    return np.array([peak_count[0].size, dat[-1, 6]])


# %%
if __name__ == "__main__":
    process = pool.Pool(processes=cpu_count() - 1)
    peaks = process.map(func=calculate_peaks, iterable=params.iterrows())

    process.close()
    process.join()

    peaks = np.array(peaks)

    params["peaks"] = peaks[:, 0]
    params["final_size"] = peaks[:, 1]

    params["peaks"] = pd.Categorical(params.peaks)
    params["final_size"] = params.final_size/N

    toc = time.time()

    print("Script time is %f" % (toc - tic))

    params.to_csv("phase_plot_maskInfluence_csv.csv")

    # %%

    pp = (gg.ggplot(data=params, mapping=gg.aes(x="w1", y="w2", fill="peaks")) +
          gg.geom_tile() +
          gg.labs(x="Social influence", y="Fear of disease", fill="Infection waves") +
          gg.scale_fill_brewer(palette="RdPu") +
          gg.theme_classic() +
          gg.facet_grid("c ~ p", labeller=gg.label_both))

    print(pp)

    pp.save(filename="phase_plot_peaks_maskInfluence.png", width=15, height=15)

    pp = (gg.ggplot(data=params[params.final_size >= 0], mapping=gg.aes(x="w1", y="w2", fill="final_size")) +
          gg.geom_tile() +
          gg.labs(x="Social influence", y="Fear of disease", fill="Final Size") +
          # gg.scale_fill_distiller(palette="RdBu", direction=1, limits=[0, 1]) +
          gg.scale_fill_continuous(limits=[0, 1]) +
          gg.theme_classic() +
          gg.facet_grid("c ~ p", labeller=gg.label_both))

    print(pp)

    pp.save(filename="phase_plot_finalSize_maskInfluence.png", width=15, height=15)
