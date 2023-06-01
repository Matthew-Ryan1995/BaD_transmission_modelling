#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:45:37 2023

@author: rya200
"""
# %% libraries
from sympy import *
x, y, z = symbols("x y z")

# %% Solving equations

# we can define equalities with Eq orabuse that solvers assume 0 equality

# solveset is the main solve with syntax
# solveset(eqn, var, domain = S.complexes)

# Examples
print(solveset(x**2 - x, x))
print(solveset(x**2 + 1, x))
print(solveset(x-x, x, domain=S.Reals))
print(solveset(sin(x)-1, x, domain=S.Reals))

# To solve a linear system of equations, use linsolve
linsolve(
    [x + y + z - 1,  # System
     x + y + 2*z - 1],
    (x, y, z)  # Vars
)

# To solve a non-linear system, use nonlinsolve
nonlinsolve(
    [
        x**2 + x, x - y  # system
    ],
    (x, y)  # vars
)

nonlinsolve(
    [
        exp(x) - sin(y),
        1/y - 3
    ],
    (x, y)
)

# %% Ploy roots

# solveset only returns each solutions onces
print(solveset(x**3 - 6*x**2 + 9*x, x))
# Use roots to get multiplicity
print(roots(x**3 - 6*x**2 + 9*x, x))

# %% Differential equations

# dsolve syntax is dsolve(DE, unknown function)

# We use dsolve, first we need to define functions
f, g = symbols("f g", cls=Function)  # Create functions

# Now f and g are unknown functions are derivatives are unevluated

# we can now write DEs
diffeq = Eq(f(x).diff(x, x) - 2*f(x).diff(x) + f(x), sin(x))

# Solve with dsolve
print(dsolve(diffeq, f(x)))

# sometimes f cannot be found explicitly, place the following into console
dsolve(f(x).diff(x)*(1-sin(f(x))) - 1, f(x))

# %%

expr = x**2 + y * x - 3
solveset(expr.subs(y, f(z)), x)
