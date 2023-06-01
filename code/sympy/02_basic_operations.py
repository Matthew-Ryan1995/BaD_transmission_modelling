#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:11:49 2023

@author: rya200
"""
# %% library
import numpy
from sympy import *

# %% Substitutions

# Can use expr.subs to input either values or new expressions, i.e.
x, y = symbols("x y")
expr = cos(x)
print(expr)

expr = expr.subs(x, sin(y))
print(expr)

# can pass in lists to do lots of replacements
# In fact, can use list comprehensions

expr = x**4 - 4*x**3 + 4*x**2 - 2*x + 3
replacements = [(x**i, y**i) for i in range(5) if i %
                2 == 0]  # Find all powers of 2
print(expr.subs(replacements))  # Replace x with y where appropriate

# %% Conversion of stings into sympy

# Use the function sympify
str_expr = "x**2 + 3*x - 1/2"
expr = sympify(str_expr)
print(expr)

# can evalue numeric expresions into floating number using evalf. Compare:
a = sqrt(8)
print(a)
print(a.evalf())

# Can go to arbitrayr precision
print(pi.evalf(20))

# Can pass in dicstionary substitutions and chop off numeric precision
one = sin(x)**2 + cos(x)**2
print((one-1).evalf(subs={x: 2}))
print((one-1).evalf(subs={x: 2}, chop=True))

# %% lambdify

# We can convert numerics into lambda functions
a = numpy.arange(10)

expr = sin(x)
f = lambdify(x,  # Variable
             expr,  # Expression
             "numpy")  # Library
print(f(a))
