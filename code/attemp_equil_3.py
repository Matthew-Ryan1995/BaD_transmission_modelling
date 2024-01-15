#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:13:57 2023

@author: rya200
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:37:45 2023

@author: rya200
"""

from BaD import *
import plotnine as gg
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import plotly.express as px
import plotly.io as io
io.renderers.default = "svg"
# io.renderers.default="browser"


# %%

model_params = dict()
model_params["p"] = 0.01
model_params["beta"] = 8
model_params["c"] = 0.01
model_params["w1"] = 0.05 * 8
model_params["w2"] = 8
model_params["w3"] = 0.01
model_params["a1"] = 0.5
model_params["a2"] = 0.5
model_params["a3"] = 0.01
model_params["gamma"] = 0.4
model_params["nu"] = 1/(8*30)

# model_params["w1"] = 1 * (model_params["a1"] +
#                           model_params["a2"] + model_params["a3"])


cust_params = dict()
cust_params["transmission"] = 8
cust_params["infectious_period"] = 1/0.4
cust_params["immune_period"] = 240
cust_params["av_lifespan"] = 0  # Turning off demography
cust_params["susc_B_efficacy"] = 0.01
cust_params["inf_B_efficacy"] = 0.01
cust_params["N_social"] = 0.5
cust_params["N_fear"] = 0.5
cust_params["B_social"] = 0.05 * 8
cust_params["B_fear"] = 8
cust_params["B_const"] = 0.01
cust_params["N_const"] = 0.01

M1 = bad(**cust_params)

tmp_params = dict(cust_params)
tmp_params_2 = dict(model_params)
# %%

# I = 0.002967715391823186

# I = model_params["nu"]/(model_params["nu"] + model_params["gamma"]) - 1e-4

# I = 0.005


def get_B_a_w(I, params):
    # assert not ((I > params["nu"]/(params["gamma"] + params["nu"])
    #              ) or (I < 0)), "invalid choice of I"
    if I < 1e-8:
        I = 0
    assert not ((I > params["nu"]/(params["gamma"] +
                params["nu"]))), "invalid choice of I"
    # print("Bad I")
    # return

    D = params["a2"] * (1-I) + params["a3"] + params["w2"] * I + params["w3"]
    C = params["w1"] - params["a1"]

    if D == 0:
        if C > 0:
            N = 0
        else:
            N = 1
    elif C == 0:
        N = (D - (params["w2"] * I + params["w3"])) / D
    else:
        N = ((C + D) - np.sqrt((C + D)**2 - 4 * C *
             (D - (params["w2"] * I + params["w3"])))) / (2 * C)
    B = 1 - N
    a = params["a1"] * (1-B) + params["a2"] * (1 - I) + params["a3"]
    w = params["w1"] * B + params["w2"] * I + params["w3"]

    return B, a, w


def get_R_S(I, params):
    R = params["gamma"]/params["nu"] * I
    S = 1 - I - R
    return R, S


def get_Ib(I, S, a, w, params):
    p = params["p"]
    c = params["c"]
    beta = params["beta"]
    g = params["gamma"]

    if (np.isclose(p, 0)) and (np.isclose(c, 0)):
        B, _, _ = get_B_a_w(I, params)
        v = params["nu"]
        R = 1-S-I
        numer = (beta*(B*(w+a+v)-w*R) + w*(w+a+v)) * I
        denom = (w+a+g+beta*I)*(a+w+v) + g
    else:
        numer = (1-c) * (beta * S - g) * I + c*w*I
        denom = (1-c) * p * beta * S + c * (a + w + g)

    ib = numer/denom

    return ib


def get_Rb(R, Ib, a, w, params):
    g = params["gamma"]
    v = params["nu"]

    Rb = (w*R + g*Ib)/(a+w+v)

    return Rb


# def get_lambda(S, I, B, a, w, params):

#     N = 1-B

#     g = params["gamma"]
#     v = params["nu"]
#     c = params["c"]

#     if c == 0:
#         lam = g * I / S
#         return lam

#     A = (a + w + g + v) * (1 - c) * S

#     C = -(w + v + a) * (a + w + g) * g * I

#     B = (a + w + g) * ((w + v) * (1-c) + a) * S - \
#         ((a + w + g) * g + v * (g + c * a)) * I + v * (a + w + g) * N

#     if (A == 0) or (c == 1):
#         lam = -C/B
#     else:
#         lam = (-B + np.sqrt(B**2 - 4 * A * C)) / (2 * A)

#     return lam


# def get_Sb(lam, S, I, B, a, w, params):

#     g = params["gamma"]
#     v = params["nu"]
#     c = params["c"]

#     if c == 0:
#         detA = (lam + a + w + v) * (a + w + g) + lam*v
#         sb = (a + w + g) * (w * S + v * B) - w * v * I
#         sb /= detA
#     else:
#         sb = (lam*S - params["gamma"] * I)/(params["c"] * lam)
#     return sb


# def get_In(lam, S, I, B, a, w, params):

#     g = params["gamma"]
#     v = params["nu"]
#     c = params["c"]

#     if c == 0:
#         detA = (lam + a + w + v) * (a + w + g) + lam*v
#         Ib = lam * (w*S + v*B) + (lam + a + w + v) * w * I
#         Ib /= detA
#         In = I - Ib
#     else:
#         In = (params["gamma"] + params["c"] * a) * \
#             I - (1 - params["c"]) * lam * S
#         In = In/(params["c"] * (a + w + params["gamma"]))

#     return In


def get_steady_states(I, params):

    B, a, w = get_B_a_w(I, params)
    N = 1-B

    if I <= 0:
        return np.array([1-B, B, 0.0, 0.0, 0.0, 0.0])

    R, S = get_R_S(I, params)

    Ib = get_Ib(I, S, a, w, params)
    Rb = get_Rb(R, Ib, a, w, params)

    Rn = R - Rb
    Sb = B - Ib - Rb
    Sn = S - Sb
    In = N - Sn - Rn

    return np.array([Sn, Sb, In, Ib, Rn, Rb])


def solve_I(i, params):

    assert "beta" in params.keys(), "define beta"

    assert "p" in params.keys(), "define p"

    B, a, w = get_B_a_w(i, params)

    ss_n = get_steady_states(i, params)

    lam = params["beta"] * (ss_n[2] + (1-params["p"]) * ss_n[3])

    res = (lam + w) * ss_n[0] - (a * ss_n[1] + params["nu"] * ss_n[4])

    return res


# init_i = model_params["nu"]/(model_params["nu"] + model_params["gamma"]) - 1e-3

# tmp = fsolve(solve_I, x0=[init_i], args=(model_params))


def find_ss(params):

    if (np.isclose(params["c"], 0)) and (np.isclose(params["p"], 0)):
        Istar = [params["nu"] * (params["beta"] - params["gamma"]) /
                 (params["beta"]*(params["gamma"] + params["nu"]))]
    else:
        init_i = params["nu"]/(params["nu"] + params["gamma"]) - 1e-3

        Istar = fsolve(solve_I, x0=[init_i], args=(params))

    if Istar[0] < 1e-8:
        Istar[0] = 0

    ss = get_steady_states(Istar[0], params)

    return ss, Istar[0]

# %%


# model_params = dict()
# model_params["p"] = 0.3
# model_params["beta"] = 8
# model_params["c"] = 0.5
# model_params["w1"] = 1.3
# model_params["w2"] = 0.5
# model_params["w3"] = 0.7
# model_params["a1"] = 0.2
# model_params["a2"] = 1.1
# model_params["a3"] = 0.9
# model_params["gamma"] = 1
# model_params["nu"] = 0.5

# cust_params = dict()
# cust_params["transmission"] = 8
# cust_params["infectious_period"] = 1/1
# cust_params["immune_period"] = 1/0.5
# cust_params["av_lifespan"] = 0  # Turning off demography
# cust_params["susc_B_efficacy"] = 0.4
# cust_params["inf_B_efficacy"] = 0.3
# cust_params["N_social"] = 0.2
# cust_params["N_fear"] = 1.1
# cust_params["B_social"] = 1.3
# cust_params["B_fear"] = 0.5
# cust_params["B_const"] = 0.7
# cust_params["N_const"] = 0.9

# M2 = bad(**cust_params)

# NN = M2.endemic_behaviour(get_res=True, save=False, I_eval=0)

# a = model_params["a1"] * NN + model_params["a2"] + model_params["a3"]
# w = model_params["w1"] * (1-NN) + model_params["w3"]

# multi_val = model_params["gamma"]*(model_params["gamma"] + a + w) / (NN * (a + model_params["gamma"] + (
#     1 - model_params["p"]) * w) + (1-NN) * (a + (1 - model_params["p"]) * (model_params["gamma"] + w)))

# beta_vals = np.arange(1.25, 3.1, step=0.1)

# res = list()

# for idxx in range(len(beta_vals)):
#     b = beta_vals[idxx] * multi_val
#     model_params["beta"] = b
#     ss, _ = find_ss(model_params)

#     # ss[ss.round(4) > 0] = 1
#     res.append(ss.round(4))

# res = np.array(res)

# plt.figure()
# plt.plot(beta_vals, res[:, 0], "green")
# plt.plot(beta_vals, res[:, 1], "black")
# plt.plot(beta_vals, res[:, 2], "red")
# plt.plot(beta_vals, res[:, 3], "magenta")
# plt.plot(beta_vals, res[:, 4], "blue")
# plt.plot(beta_vals, res[:, 5], "cyan")
# plt.show()


# ww = np.arange(0.1, 4.1, step=0.1)
# bb = np.arange(0.1, 10.1, step=0.1)

# x, y = np.meshgrid(bb, ww)

# xx = np.array(np.meshgrid(bb, ww)).reshape(2, len(bb) * len(ww)).T

# %%

MM = bad(**tmp_params)
MM.run([1-1e-6, 0, 1e-6, 0, 0, 0], 0, 600, 1)
ss, _ = find_ss(tmp_params_2)

plt.figure()
plt.plot(MM.t_range, MM.results[:, 0], "green", label="Sn")
plt.plot([MM.t_range[0], MM.t_range[-1]], [ss[0], ss[0]],
         "green", linestyle=":", label="Sn")
plt.plot(MM.t_range, MM.results[:, 1], "black", label="Sb")
plt.plot([MM.t_range[0], MM.t_range[-1]], [ss[1], ss[1]],
         "black", linestyle=":", label="Sb")
plt.show()
plt.figure()
plt.plot(MM.t_range, MM.results[:, 2], "red", label="In")
plt.plot([MM.t_range[0], MM.t_range[-1]], [ss[2], ss[2]],
         "red", linestyle=":", label="In")
plt.plot(MM.t_range, MM.results[:, 3], "magenta", label="Ib")
plt.plot([MM.t_range[0], MM.t_range[-1]], [ss[3], ss[3]],
         "magenta", linestyle=":", label="Ib")
plt.show()
plt.figure()
plt.plot(MM.t_range, MM.results[:, 4], "blue", label="Rn")
plt.plot([MM.t_range[0], MM.t_range[-1]], [ss[4], ss[4]],
         "blue", linestyle=":", label="Rn")
plt.plot(MM.t_range, MM.results[:, 5], "cyan", label="Rb")
plt.plot([MM.t_range[0], MM.t_range[-1]], [ss[5], ss[5]],
         "cyan", linestyle=":", label="Rb")
plt.legend(loc=[1, 0.])
plt.show()

# # %%
# Istar = [tmp_params_2["nu"] * (tmp_params_2["beta"] - tmp_params_2["gamma"]) /
#          (tmp_params_2["beta"]*(tmp_params_2["gamma"] + tmp_params_2["nu"]))]
# plt.figure()
# plt.plot(MM.t_range, MM.results[:, [2, 3]].sum(1))
# plt.plot([MM.t_range[0], MM.t_range[-1]], [Istar[0], Istar[0]], "k:")
# plt.show()


# if (np.isclose(tmp_params_2["c"], 0)) and (np.isclose(tmp_params_2["p"], 0)):
#     print("yes")
# else:
#     print("no")
# ss_results = list()
# R0 = list()

# # for w in xx[:, 1]:
# # for b in xx[:, 0]:
# for idxx in range(len(xx)):
#     # print(f"{idxx}")
#     w = xx[idxx, 1] * (model_params["a1"] +
#                        model_params["a2"] + model_params["a3"])
#     b = xx[idxx, 0] * model_params["gamma"]
#     model_params["w1"] = w
#     model_params["beta"] = b
#     cust_params["B_social"] = w
#     cust_params["transmission"] = b
#     ss, _ = find_ss(model_params)
#     M1.update_params(**cust_params)
#     R0.append(M1.Rzero())

#     # ss[ss.round(4) > 0] = 1

#     ss_results.append(ss.round(4))
# # %%


# def calc_B(X):
#     xx = X[1]
#     B = xx[[1, 3, 5]].sum()
#     return B


# # BB = list(map(calc_B, enumerate(ss_results)))


# def calc_I(X):
#     xx = X[1]
#     B = xx[[2, 3]].sum()
#     return B


# BB = list(map(calc_B, enumerate(ss_results)))
# II = list(map(calc_I, enumerate(ss_results)))

# states_2 = np.zeros(len(BB))

# for idx in range(len(states_2)):
#     if BB[idx] > 0:
#         states_2[idx] += 1
#     if II[idx] > 0:
#         states_2[idx] += 2

# # %%


# plt.figure()
# plt.title("Steady state of behaviour")
# plt.contourf(x, y, np.array(BB).reshape(x.shape),  cmap=plt.cm.Blues)
# plt.colorbar()
# plt.xlabel("Epidemic R0")
# plt.ylabel("Behaviour R0")
# plt.show()

# plt.figure()
# plt.title("Steady state of Infection")
# plt.contourf(x, y, np.array(II).reshape(x.shape),  cmap=plt.cm.Reds)
# plt.colorbar()
# plt.xlabel("Epidemic R0")
# plt.ylabel("Behaviour R0")
# plt.show()

# plt.figure()
# plt.title("Behaviour affected R0")
# plt.contourf(x, y, np.array(R0).reshape(x.shape),  cmap=plt.cm.Greens)
# plt.colorbar()
# plt.xlabel("Epidemic R0")
# plt.ylabel("Behaviour R0")
# plt.show()


# # %%

# plt.figure()
# plt.title("Steady state of Infection")
# # plt.contourf(x, y, np.array(II).reshape(x.shape),  cmap=plt.cm.Reds)
# plt.imshow(np.array(II).reshape(x.shape),  cmap=plt.cm.Reds,
#            origin='lower',
#            extent=[x.min(), x.max(), y.min(), y.max()],
#            aspect="auto")
# plt.colorbar()
# plt.xlabel("Epidemic R0")
# plt.ylabel("Behaviour R0")
# plt.show()
