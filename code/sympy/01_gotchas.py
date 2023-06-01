#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:42:02 2023

@author: rya200
"""

# %% import library

from sympy import *

# We need to define out symbols before we start
# Also note that python variables do not need to match with sympy variables
a, b = symbols("b a")
print(a)

# %% The difference between symbols and variables

# Python does not back trace the changes in variables
x = symbols("x")
expr = x + 1
x = 2
print(expr)
print(x)

# We can evaluate expressions using the expr.subs function
x = symbols("x")
expr = x + 1
print(expr.subs(x, 2))

# %% Equalities
# Cannnot use = or ==
# Use the Eq function

expr1 = Eq(expr, 4)

# To test if two symbolic expressions are equal, we can use the simplify function or the equals function
# Easier to test if a-b = 0
a = (x + 1)**2
b = x**2 + 2*x + 1

# Does symbolic manipulation to see if things cancel
# Not always gonna work
print(simplify(a - b))

a = cos(x)**2 - sin(x)**2
b = cos(2*x)

# Tests random points to see if equal
print(a.equals(b))

# %% Rationals

# Sympy first evalues integers, then converts to sympy
# To ensure we have pretty symbols, we need to specify rationals explicitly.
# Compare

expr1 = x + 1/2
expr2 = x + Rational(1, 2)

print(expr1)
print(expr2)
