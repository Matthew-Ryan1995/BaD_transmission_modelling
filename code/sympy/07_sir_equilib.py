#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:27:56 2023

This file runs an equilibrium calculation for my model without behavioural transitions

@author: rya200
"""
# %% libraries
from sympy import *

# %% Symbols
S, I, R = symbols("S I R", positive=True)
beta, gamma, nu = symbols("B g v", positive=True)

# %% DEs

Sdot = -beta*S*I + nu*R
Idot = beta*S*I - gamma*I
Rdot = gamma*I - nu*R
N = S+I+R-1

# %% solutions

sols = solve(
    [
        Sdot,
        Idot,
        N
    ],
    (S, I, R),
    dict=True
)

# %% Something harder
Sm, Sn, Im, In, Rm, Rn = symbols("Sm Sn Im In Rm Rn", positive=True)
beta1, gamma, nu = symbols("B g v", positive=True)
p, c = symbols("p c", positive=True)
# works when p = 1 or c = 0
p = 1
beta2 = beta1*(1-p)

delta = symbols("D", positive=True)

# %% Eqn

Sndot = -(beta1*In + beta2*Im) * Sn + nu * Rn
Smdot = -(beta1*In + beta2*Im) * Sm * (1-c) + nu * Rm
Indot = (beta1*In + beta2*Im) * Sn - gamma * In
Imdot = (beta1*In + beta2*Im) * Sm * (1-c) - gamma * Im
Rndot = gamma * In - nu * Rn
Rmdot = gamma * Im - nu * Rm
N = delta - (Sn + In + Rn)
M = 1 - delta - (Sm + Im + Rm)

eqn_set = [Sndot, Smdot, Indot, Imdot, N, M]

sols = solve(eqn_set, (Sn, Sm, In, Im, Rn, Rm), dict=True)
