#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:35:59 2023

@author: rya200
"""
# %% Libraries
from sympy import *

# %% Create symbols

N, I = symbols("N I", positive=True)
w1, w2, w3 = symbols("w1:4", positive=True)
a1, a2, a3 = symbols("a1:4", positive=True)

M = 1-N

# %% Create bits and bobs

a = a1*N + a2*(1-I) + a3
w = w1*M + w2*I + w3

N_change = -w*N + a*M

dfe_N = N_change.subs(I, 0)

# %% Solve equation

full_sol = solve(dfe_N, N)

interval_sol = solve(dfe_N, N, Interval(0, 1))

quad_deter = a1-a2-a3-w1-w3 - full_sol[0] * 2*(a1-w1)
quad_deter = quad_deter**2

my_deter = (w1 - a1 + w3 + a2 + a3)**2 - 4 * (w1 - a1) * (a2 + a3)

simplify(quad_deter - my_deter)

# %% a1 = w1

dfe_N_a1w1 = dfe_N.subs(a1, w1)

full_sol_a1w1 = solve(dfe_N_a1w1, N)

# %% w2+a2+a3 = 0

dfe_N_zeros = dfe_N.subs([(w3, 0), (a2, 0), (a3, 0)])

full_sol_zeros = solve(dfe_N_zeros, N)
# %% w3+a2+a3 = 0

dfe_N_zeros_a1w1 = dfe_N.subs([(w3, 0), (a2, 0), (a3, 0), (a1, w1)])

full_sol_zeros_a1w1 = solve(dfe_N_zeros, N)

# %% Set of eqn?
N, I, M = symbols("N I M", positive=True)
w1, w2, w3 = symbols("w1:4", positive=True)
a1, a2, a3 = symbols("a1:4", positive=True)


a = a1*N + a2*(1-I) + a3
w = w1*M + w2*I + w3

N_change = -w*N + a*M

dfe_N = N_change  # .subs([(I, 0)])

eqnset = [dfe_N, N + M - 1]
ANS = solve(eqnset, (M, N))
