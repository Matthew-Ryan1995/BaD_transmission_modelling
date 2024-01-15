#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:27:33 2023

@author: rya200
"""

# %% Packages
from sympy import *

# %% Set up symbols

# Behaviour strata
Sb, Ib, Rb = symbols("Sb Ib Rb")

# Non-beahviour strata
Sn, In, Rn = symbols("Sn In Rn")

# infection Parameters
b, g, v = symbols("b g v")
c, p = symbols("c p")

# Behaviour parameters
a1, a2, a3, w1, w2, w3 = symbols("a1:4 w1:4")

# Transitions between strata

alpha = a1 * (Sn + In + Rn) + a2 * (1 - (In + Ib)) + a3
omega = w1 * (Sb + Ib + Rb) + w2 * (In + Ib) + w3

# %% derivatives

SnDot = -b * (In + (1-p) * Ib) * Sn + alpha * Sb - omega * Sn + v * Rn
SbDot = -b * (1-c) * (In + (1-p) * Ib) * Sb - alpha * Sb + omega * Sn + v * Rb
InDot = b * (In + (1-p) * Ib) * Sn + alpha * Ib - omega * In - g * In
IbDot = b * (1-c) * (In + (1-p) * Ib) * Sb - alpha * Ib + omega * In - g * Ib
RnDot = alpha * Rb - omega * Rn + g * In - v * Rn
RbDot = - alpha * Rb + omega * Rn + g * Ib - v * Rb


# %% Evaluation parameters

D = symbols("D")  # Proportion no mask in steady state

eval_symbs = [(Sn, D), (Sb, 1-D), (In, 0), (Ib, 0), (Rn, 0), (Rb, 0)]


symb_list = [Sn, Sb, In, Ib, Rn, Rb]
deriv_list = [SnDot, SbDot, InDot, IbDot, RnDot, RbDot]

# %% Jacobian matrix

J = zeros(6, 6)

linearised_eqn = []

for j in range(len(symb_list)):
    x = symb_list[j]
    for i in range(len(deriv_list)):
        F = deriv_list[i]
        J[i, j] = F.diff(x)  # .subs(eval_symbs).collect(D)

    # linearised_eqn.append(SnDot.diff(x).subs(eval_symbs).collect(D))

# %% J at DFE

res = J.subs(eval_symbs)

dat_vec = [Sn, Sb, In, Ib, Rn, Rb]

Sn1 = res[0, 0] * dat_vec[0]
Sb1 = res[1, 0] * dat_vec[0]
In1 = res[2, 0] * dat_vec[0]
Ib1 = res[3, 0] * dat_vec[0]
Rn1 = res[4, 0] * dat_vec[0]
Rb1 = res[5, 0] * dat_vec[0]

for j in range(len(dat_vec)):
    k = j+1
    if k == 6:
        break
    Sn1 += res[0, k]*dat_vec[k]
    Sb1 += res[1, k]*dat_vec[k]
    In1 += res[2, k]*dat_vec[k]
    Ib1 += res[3, k]*dat_vec[k]
    Rn1 += res[4, k]*dat_vec[k]
    Rb1 += res[5, k]*dat_vec[k]

(Sn1 + Sb1).simplify().collect(Sb).collect(Sn)
(In1 + Ib1).simplify().collect(Ib).collect(In)
(Rn1 + Rb1).simplify().collect(Rb).collect(Rn)

# %%
f1, f2, f3, f4, f5, f6 = symbols("f1:7", cls=Function)
t = symbols("t")
pp, cc = symbols("pBar cBar")

Sn_reduce = Sn1.subs([(Rn, 0), (Rb, 0), (D, Sn), (Sb, 1-Sn),
                      (Sn, f1(t)), (In, f2(t)), (Ib, f3(t))])
In_reduce = In1.subs([(Rn, 0), (Rb, 0), (D, Sn), (Sb, 1-Sn),
                      (Sn, f1(t)), (In, f2(t)), (Ib, f3(t))])
Ib_reduce = Ib1.subs([(Rn, 0), (Rb, 0), (D, Sn), (Sb, 1-Sn),
                      (Sn, f1(t)), (In, f2(t)), (Ib, f3(t))])

system_1 = Matrix([Sn_reduce, In_reduce, Ib_reduce])

f_vec = Matrix([f1(t), f2(t), f3(t)])


ODE_eqns = f_vec.diff(t) - system_1

system_2 = res[[2, 3], [2, 3]].subs(
    [(Rn, 0), (Rb, 0), (D, Sn), (Sb, 1-Sn),
     (1-p, pp), (1-c, cc)
     # (a1, 0), (a2, 0), (w1, 0), (w2, 0),
     ])

for i in range(system_2.shape[0]):
    for j in range(system_2.shape[1]):
        system_2[i, j] = system_2[i, j].expand().collect(Sn).collect(b)
# %%
sols = system_2.eigenvects()

# sols = dsolve(
#     [Eq(Derivative(f1(t), t), system_1[0]),
#      Eq(Derivative(f2(t), t), system_1[1]),
#      Eq(Derivative(f3(t), t), system_1[2])],
#     [f1(t), f2(t), f3(t)]
#     )

# Sols = dsolve(list(ODE_eqns), list(f_vec))

# %% Solve the linearised system?

f1, f2, f3, f4, f5, f6 = symbols("f1:7", cls=Function)
t = symbols("t")

f_vec = Matrix([f1(t), f2(t), f3(t), f4(t), f5(t), f6(t)])

Df = res * f_vec

ODE_eqns = f_vec.diff(t) - Df

# Sols = dsolve(list(ODE_eqns), list(f_vec)) # Mo simply solution to this system methinks

# %% Steady State for S, I, R

SDot = SnDot + SbDot
IDot = InDot + IbDot
RDot = RnDot + RbDot

constraint_1 = Eq(Sn+In+Rn, D)
constraint_2 = Eq(Sb+Ib+Rb, 1-D)
constraint_3 = Eq(Sb+Ib+Rb+In+Sn+Rn, 1)

constraint_subs = [
    (p, 1), (c, 0),
    (a1, 0),
    (a2, 0),
    (a3, 0),
    (w1, 0),
    (w2, 0),
    # (w3, 0)
]

# sol = solve([SDot, IDot, RDot, constraint_1,
#             constraint_2, constraint_3], symb_list)

sol2 = solve([SnDot.subs(constraint_subs), SbDot.subs(constraint_subs),
              InDot.subs(constraint_subs), IbDot.subs(constraint_subs),
              RnDot.subs(constraint_subs), RbDot.subs(constraint_subs),
              constraint_3], symb_list)
