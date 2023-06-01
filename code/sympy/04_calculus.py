#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:20:20 2023

@author: rya200
"""
# %% libraries
from sympy import *
x, y, z = symbols("x y z")

# %% derivatives

# Derivatives are taken using the diff function
# Pass in expression and variables to be differentiated with respect to

expr = exp(x) * cos(x)**2 - x*y
print(expr.diff(x, 1))  # First deriv
print(expr.diff(x, 2))  # Second deriv

# Can do multi functions
# tp get d2/dxdy
print(expr.diff(x, y))

# Unevaluated derivatives:
dd = Derivative(expr, x, y)

# To evaluate:
dd.doit()

# Note, can define arbitrary derivatives using symbols too

# %% Integrals
# use the integrate function
# order pased in is integrate(func, (var, lower_lim, upper_lim))
print(integrate(exp(-x), (x, 0, oo)))

# Note oo is infinity symbol
# Can also do mutlivariate integrals

print(integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo)))

# Unevaluated intergarls with Integrate and doit
# Can return piecewise results
integ = Integral(x**y*exp(-x), (x, 0, oo))
# Type into console

# %% limits

# Compute function limits using syntax limit(func, var, lim_point)
# Use limit instead of subs when there is a singularity
print((x**2/exp(x)).subs(x, oo))
print(limit(x**2/exp(x), x, oo))

# Also has un-eval counterpart Limit
# can do one-sided limits using + or - as fourth argument

# If I want to explore finite differences, return to this tutorial:
# https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html
