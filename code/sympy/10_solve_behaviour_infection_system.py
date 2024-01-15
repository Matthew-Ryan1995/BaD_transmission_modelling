#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:04:12 2023

Solve the system for the behaviour B when infections are approximately constant

Sympy does not give an exact answer, but wolfram does

@author: rya200
"""
# %% Packages
from sympy import *

# %% Define symbols

B = Function("B")
I = Function("I")
t, a0, a1, a2, a3 = symbols("t a0:4")

# Bdot = a0 + a1*B(t) + a2*B(t)**2
Bdot = a3 * B(t) + a2 * I(t) * B(t) + a1 * I(t) + a0

# %% Solve the DE basic

# B_sol = dsolve(Eq(Derivative(B(t), t), Bdot), B(t))

# %%

# k = sqrt(-1/(4*a0*a2-a1**2))

# m_p = (-4*a0*a2*k + a1**2 * k + a1)/(2*a2)
# m_m = (4*a0*a2*k - a1**2 * k + a1)/(2*a2)

# C1 = m_p * log(k) / m_m

# B_tmp = (m_m * (C1 - t) - m_p * log(k))/(C1-t)

# %%

# sol = B_tmp.diff(t) - (a0 + a1*B_tmp + a2*B_tmp**2)

# %% Result from wolfram

x = symbols("x")

a2, a3 = symbols("a2:4")
w3 = symbols("w3")

eqn = (x + w3 + a2 + a3)**2 - 4*x*w3

simplify(eqn, x)
