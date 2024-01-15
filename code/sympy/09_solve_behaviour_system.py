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
t, a0, a1, a2 = symbols("t a0:3")

Bdot = a0 + a1*B(t) + a2*B(t)**2

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

k2 = sqrt(4*a0*a2 - a1**2)
c1 = symbols("c1")

B2 = (k2 * tan((k2 * c1 + k2 * t)/2) - a1)/(2*a2)

sol = B2.diff(t) - (a0 + a1*B2 + a2*B2**2)

# %% Sub in a0, a1, a2

e, w1, w2, w3, b1, b2, b3 = symbols("e w1:4 b1:4", postive=True)

a0_exact = w2*e + w3
a1_exact = w1 - w2*e - w3 - b1 - b2*(1-e) - b3
a2_exact = b1-w1

B_full_sol = B2.subs([(a0, a0_exact), (a1, a1_exact), (a2, a2_exact)])
B_full_sol = simplify(B_full_sol)
