#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:37:05 2023

@author: rya200
"""

# %%
from scipy.integrate import solve_ivp
import numpy as np

# %%

# w1 = 8
# R0 = 2
# gamma = 0.4

# cust_params = dict()
# cust_params["transmission"] = R0*gamma
# cust_params["infectious_period"] = 1/gamma
# cust_params["immune_period"] = 240
# cust_params["av_lifespan"] = 0  # Turning off demography
# cust_params["susc_B_efficacy"] = 0.8
# cust_params["inf_B_efficacy"] = 0.4
# cust_params["N_social"] = 0.5
# cust_params["N_fear"] = 0
# cust_params["B_social"] = 0.05 * w1
# cust_params["B_fear"] = w1
# cust_params["B_const"] = 0.01
# cust_params["N_const"] = 0.01
# FS = M1.results[-1,[0, 2, 4, 1, 3, 5]]

model_params = dict()
model_params["p"] = 0.4
model_params["c"] = 0.8
model_params["w1"] = 0.05 * 8
model_params["w2"] = 8
model_params["w3"] = 0.01
model_params["a1"] = 0.5
model_params["a2"] = 0.
model_params["a3"] = 0.01
model_params["gamma"] = 0.4
model_params["nu"] = 1/(8*30)


def get_B_a_w(I, params):
    if (I > 1) or (I < 0):
        print("Bad I")
        return

    D = params["a2"] * (1-I) + params["a3"] + params["w2"] * I + params["w3"]
    C = params["w1"] - params["a1"]

    if C == 0:
        N = (D - (params["w2"] * I + params["w3"])) / D
    else:
        N = ((C + D) - np.sqrt((C + D)**2 - 4 * C *
             (D - (params["w2"] * I + params["w3"])))) / (2 * C)
    B = 1 - N
    a = params["a1"] * (1-B) + params["a2"] * (1 - I) + params["a3"]
    w = params["w1"] * B + params["w2"] * I + params["w3"]

    return B, a, w


def create_linear_system(l, a, w, params):

    c = params["c"]
    p = params["p"]
    g = params["gamma"]
    v = params["nu"]

    M = np.matrix(
        # Sn In Rn  Sb Ib Rb
        [[-l-w, 0, v, a, 0, 0],  # Sn
         [l, -w-g, 0, 0, a, 0],  # In
         [0, g, -w-v, 0, 0, a],  # Rn
         [w, 0, 0, -l*(1-c) - a, 0, v],  # Sb
         [0, w, 0, l*(1-c), -a-g, 0],  # Ib
         [0, 0, w, 0, g, -a-v]]  # Rb
    )

    # M[-1, :] = [1, 1, 1, 0, 0, 0]
    M[-1, :] = 1
    # M[-2, :] = [0, 1, 0, 0, 1, 0]
    # M[-3, :] = [0, 0, 0, 1, 1, 1]

    return M


Istar = 0.002967715391823186

B, a, w = get_B_a_w(Istar, model_params)

lam_guess = (Istar * (1-B) + (1-model_params["p"]) * Istar * B) * (model_params["gamma"] + model_params["nu"]
                                                                   ) / model_params["nu"] + model_params["gamma"]

lam_guess = 0.002183611285462691
lam_guess = 0.00219
# lam_guess = 0.001

A = create_linear_system(lam_guess, a, w, model_params)

ss = np.linalg.solve(A, np.array([0, 0, 0, 0, 0, 1]))

beta = lam_guess/(ss[1] + (1-model_params["p"]) * ss[4])

print(f"I is {(ss[1] + ss[4]).round(4)} should be {Istar}")
print(f"B is {ss[3:6].sum().round(4)} should be {B.round(4)}")

# %%

beta = 5 * model_params["gamma"]

II = model_params["nu"] * (beta - model_params["gamma"]) / \
    (model_params["gamma"] + model_params["nu"])

bb = II * (model_params["gamma"] + model_params["nu"]
           ) / model_params["nu"] + model_params["gamma"]
# I think lambda is not the correct value to choose, but what is?

# %%

i = 0.05

lam = 0.9
nu = 1/240
gamma = 0.4

beta = lam/i

M = np.mat(
    [[-lam, 0, nu],
     [beta * i, -gamma, 0],
     [1, 1, 1]]
)

v = np.array([0, i, 1])

x = np.matmul(np.linalg.inv(M), v)


def test_fn(t, PP):
    Y = np.zeros(3)

    Y[0] = -beta * PP[0] * PP[1] + nu * PP[2]
    Y[1] = beta * PP[0] * PP[1] - gamma * PP[1]
    Y[2] = gamma * PP[1] - nu * PP[2]
    return Y


tmp = solve_ivp(test_fn, [0, 100000], [1-1e-3, 1e-3, 0])
