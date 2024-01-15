#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:44:53 2023

@author: rya200
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:08:25 2023

Shape discontiunity when w2 = 8.68, so w1 = 0.434
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
import matplotlib.pyplot as plt
import seaborn as sns
import time
working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %% Set up

tic = time.time()

TS = 1.0
ND = 300.0

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

# w2 = a2 = np.arange(0, 0.5, step=0.025)
w2 = a2 = np.arange(0, 20, step=0.1)
# w2 = 0.05 * a2


params = np.array(np.meshgrid(c, p, w2, a2)).reshape(
    4, len(c) * len(p) * len(a2) * len(w2)).T

params = pd.DataFrame(params, columns=["c", "p", "w2", "a2"])
params = params.round(3)

params.reset_index()
# params.w2 = 0.05 * params.w2


def event(t, y):
    if t > 10:
        ans = y[2] + y[3] - 1
    else:
        ans = 1
    return ans


event.terminal = True

p = 0.8


def get_final_size(w2):
    new_params = dict()

    new_params["transmission"] = 5 * 0.4
    new_params["susc_mask_efficacy"] = p
    new_params["inf_mask_efficacy"] = p
    new_params["mask_social"] = w2 * 0.05
    new_params["mask_fear"] = w2
    new_params["infectious_period"] = 1/0.4
    # cust_params["immune_period"] = 0
    new_params["av_lifespan"] = 0
    new_params["nomask_social"] = 0.5
    new_params["nomask_fear"] = 0.

    model = msir(**new_params)

    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=init_cond,
                                  t_eval=t_range,
                                  events=[event])
    dat = RES.y.T

    peak_count = spi.signal.find_peaks(
        dat[:, 2] + dat[:, 3], height=(1, None), prominence=(100, None))

    return [dat[-1, 6], np.mean(np.sum(dat[:, 0:5:2], axis=1))/N, peak_count[0].size]


if __name__ == "__main__":
    process = pool.Pool(processes=cpu_count() - 1)
    peaks = process.map(func=get_final_size, iterable=w2)

    process.close()
    process.join()

    peaks = np.array(peaks)

    # params["peaks"] = peaks[:, 0]
    # params["final_size"] = peaks  # [:, 0]

    # params["peaks"] = pd.Categorical(params.peaks)
    # params["final_size"] = params.final_size/N

    toc = time.time()

    print("Script time is %f" % (toc - tic))

    # params.to_csv("phase_plot_maskInfluence_csv_fear_explore.csv")

    # %%

    plt.figure(figsize=[8, 8*0.618], dpi=600)
    plt.title("Final size by fear of disease")
    plt.xlabel("w2")
    plt.ylabel("Final size (proportion)")
    plt.plot(w2, peaks[:, 0]/N, ".")
    plt.savefig(fname="img/final_size_by_w2.png")
    plt.show()

    plt.figure()
    plt.plot(peaks[:, 1], peaks[:, 0]/N, c="green")
    plt.show()

    plt.figure(figsize=[8, 8*0.618], dpi=600)
    sns.boxplot(x=peaks[:, 2], y=peaks[:, 0]/N, hue=w2 <= 8.7)
    plt.legend(title="w2 <= 8.7")
    plt.xlabel("Number of waves")
    plt.ylabel("Proportion final size")
    plt.title("Final size by number of epidemic waves")
    plt.savefig(fname="img/final_size_by_peaks.png")
    plt.show()
    # pp = (gg.ggplot(data=params, mapping=gg.aes(x="w2", y="a2", fill="peaks")) +
    #       gg.geom_tile() +
    #       gg.labs(x="w2", y="a2", fill="Infection waves") +
    #       gg.scale_fill_brewer(palette="RdPu") +
    #       gg.theme_classic() +
    #       gg.facet_grid("c ~ p", labeller=gg.label_both))

    # print(pp)

    # pp.save(filename="phase_plot_peaks_maskInfluence_sfear_explore.png",
    #         width=15, height=15)

    # pp = (gg.ggplot(data=params[params.final_size >= 0], mapping=gg.aes(x="w2", y="a2", fill="final_size")) +
    #       gg.geom_tile() +
    #       gg.labs(x="w2", y="a2", fill="Final Size") +
    #       # gg.scale_fill_distiller(palette="RdBu", direction=1, limits=[0, 1]) +
    #       gg.scale_fill_continuous(limits=[0, 1]) +
    #       gg.theme_classic() +
    #       gg.facet_grid("c ~ p", labeller=gg.label_both))

    # print(pp)

    # pp.save(filename="phase_plot_finalSize_maskInfluence_fear_explore.png",
    #         width=15, height=15)
