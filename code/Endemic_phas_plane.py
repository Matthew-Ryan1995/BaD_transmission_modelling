#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:15:51 2023

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

beta_list = np.arange(0, 10, step=0.1)

s_vals = []
i_vals = []


for b in beta_list:
    m_params["transmission"] = b
    ss, _ = find_ss(m_params)
    s_vals.append(ss[[0, 1]].sum())
    i_vals.append(ss[[2, 3]].sum())

# %%

plt.figure()
plt.plot(s_vals, i_vals)
plt.ylabel("I")
plt.xlabel("S")
plt.show()
