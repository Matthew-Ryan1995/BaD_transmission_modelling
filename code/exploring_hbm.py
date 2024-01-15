#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:25:17 2023

Need a reasonable odds ratio

@author: rya200
"""

# %% package
import numpy as np
import plotnine as gg
import pandas as pd

# %% params

disease_prev = np.arange(start=0, stop=1.01, step=1e-2)
social_influence = np.arange(start=0, stop=1.1, step=0.1)
social_influence = np.round(social_influence, 1)

X = np.array(np.meshgrid(disease_prev, social_influence)).reshape(
    2, len(disease_prev) * len(social_influence)).T

b0 = 100
b1 = -99
b4 = -0.5*b1

logit_prob = b0 + b1 * X[:, 0] + b4 * X[:, 1]

# prob = np.exp(logit_prob)/(1 + np.exp(logit_prob)) # If logistic
prob = 1/logit_prob

dat = pd.DataFrame(X, columns=["d", "s"])
dat["prob"] = prob
dat["OR"] = np.exp(logit_prob)
dat.s = pd.Categorical(dat.s)

# %%

pp = (gg.ggplot(data=dat, mapping=gg.aes(x="d", y="prob", colour="s")) +
      gg.geom_line())

print(pp)

# pp = (gg.ggplot(data=dat[dat.d < 1.1], mapping=gg.aes(x="d", y="OR", colour="s")) +
#       gg.geom_line())

# print(pp)
