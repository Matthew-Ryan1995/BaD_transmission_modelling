#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 13:42:03 2023

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

inf_per = 7
immune_period = 240
R0d = 5.4

m_params = dict()
m_params["transmission"] = R0d/inf_per
m_params["infectious_period"] = inf_per
m_params["immune_period"] = immune_period
m_params["av_lifespan"] = 0  # Turning off demography
m_params["susc_B_efficacy"] = 1.
m_params["inf_B_efficacy"] = 1.
m_params["N_social"] = 0.2
m_params["B_social"] = .3
m_params["B_fear"] = 0.7
m_params["B_const"] = 0.3
m_params["N_const"] = 0.9


# %%

Sb = 1e-3
In = 1e-3
Ib = Rn = Rb = 0.0
Sn = 1 - Sb - Ib - In - Rb - Rn

PP = [Sn, Sb, In, Ib, Rn, Rb]

t_start, t_end = [0, 400]

M = bad(**m_params)
M.run(PP, t_start, t_end)

plt.figure()
plt.title("I")
plt.plot(M.get_I())
plt.show()

plt.figure()
plt.title("B")
plt.plot(M.get_B())
plt.show()

plt.figure()
plt.title("Method 1")
plt.plot(M.NGM(get_res=True))
plt.show()

plt.figure()
plt.title("Method 2")
plt.plot(M.NGM(get_res=True, orig=True))
plt.show()


# %%

plt.figure()
plt.title("Compare")
plt.plot(M.NGM(get_res=True), label="bad, full protect")
# M.update_params(**{"B_fear": 1.4})
# M.run(PP, t_start, t_end)
# plt.plot(M.NGM(get_res=True), label="Doubled w2")
# M.update_params(**{"B_fear": 2.1})
# M.run(PP, t_start, t_end)
# plt.plot(M.NGM(get_res=True), label="Tripled w2")
M.update_params(**{"susc_B_efficacy": 0, "inf_B_efficacy": 0})
M.run(PP, t_start, t_end)
plt.plot(M.NGM(get_res=True), label="SIRS")
plt.legend()
plt.show()
