#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:49:40 2023

@author: rya200
"""
# %% libraries
from sympy import *

x, y, z = symbols("x y z")

# %% Simplify

# Simplify is an all-rounder that tries a bunch.  Likely much slower than if we know where we are going
print("Simplify")
print(simplify(sin(x)**2 + cos(x)**2))
print(simplify((x**3 + x**2 - x - 1) / (x**2 + 2*x + 1)))
print(simplify(gamma(x)/gamma(x-2)))

# %% Polys and rationals

# expand breaks apart and expands parenthases
print("Expand")
print(expand((x + 1)**2))
print(expand((x + 1) * (x - 3)))

# Expand can simplify due to cancelling
print(expand((x + 1) * (x - 2) - (x - 1) * x))

# Factor facotrs equations over the rationals into irriducilbes
print("Factor")

print(factor(x**3 - x**2 + x - 1))
print(factor(x**2 + 9))  # Notice how this does not factor

# factor_list will return thelist of facotrs
print(factor_list(x**3 - x**2 + x - 1))

# Note that the epxressions need not be polynomials, can be sin/cos/etc

# collect collects like terms!
print("Collect")
expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
print(expr)
print(collect(expr,  # What to collect like terms in
              x))  # What to collect over!

# Can use collect ot ofind the coefficients of certain terms!
print(collect(expr, x).coeff(x, 2))

# Cancel will simplify rational functions
print("cancel")

expr = 1/x + (3*x/2) - 2/(x - 4)
print(expr)
print(cancel(expr))

# apart performs a partial fraction decomposition of a rational function
print("apart")
print(
    apart(
        (4*x**3 + 21*x**2 + 10*x + 12) / (x**4 + 5*x**3 + 5*x**2 + 4*x)
    )
)

# %% Trig simplifcations

# Trigsimp simplifies based on trig identities
print("trigsimp")
print(trigsimp(cos(x)*sin(2*y) + sin(x) * cos(2*y)))
# expand_trig does the opposite
print(expand_trig(sin(x + 2 * y)))

# %% Powers

# Coolest point, we can put conditions on our symbols
# Without specifying, assumes symbols are complex
# for reference:
# 1: x^a x^b = x^(a+b)
# 2: x^ay^a = (xy)^a
# 3:  (x^a)^b = x^(ab)

x, y = symbols("x y", positive=True)
a, b = symbols("a b", real=True)
z, t, c = symbols("z t c")

# powsimp applies 1 and 2 from left to right
print("powsimp")
print(powsimp(x**a * x**b))
print(powsimp(x**a * y ** a))

# NOTE, will not work if operation not valid, i.e.
print(powsimp(t**c * z**c))
# But can be forced
print(powsimp(t**c * z**c, force=True))

# Can do all of these explicitly with:
# expand_power_base (1)
# expand_power_exp (2)
# powdenest (3)

# %% Special funtions

# A short list of special functions
# factorial
# binomial
# gamma
# hyper - generalised hypergeometric function

# A cool function is rewrite:
print(tan(x).rewrite(cos))
print(factorial(x).rewrite(gamma))

# %% Example: continued fractions

def list_to_frac(l):
    expr = Integer(0)
    for i in reversed(l[1:]):
        expr += i
        expr = 1/expr
    return l[0] + expr

a0, a1, a2, a3, a4 = symbols("a0:5")

import random
l = list(symbols('a0:5'))
random.shuffle(l)
orig_frac = frac = cancel(list_to_frac(l))
del l

# something with apart_list maybe?