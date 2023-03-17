#!/usr/bin/env python

####################################################################
###    This is the PYTHON version of program 2.1 from page 19 of   #
### "Modeling Infectious Disease in humans and animals"            #
### by Keeling & Rohani.										   #
###																   #
### It is the simple SIR epidemic without births or deaths.        #
####################################################################

###################################
### Written by Ilias Soumpasis    #
### ilias.soumpasis@ucd.ie (work) #
### ilias.soumpasis@gmail.com	  #
###################################
# %% Packages
import scipy.integrate as spi
import numpy as np
# import pylab as pl
import matplotlib.pyplot as plt
import json
import os


working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)
# %% Parameters and inital conditions

filename = "data/parameter_ranges.json"

# Load in the data from a json file
with open(filename) as json_file:
    pars = json.load(json_file)
# Replace all the infromation in the json with the expected value
for key, value in pars.items():
    pars[key] = value["exp"]

# Enter custom params
cust_params = dict()
# cust_params["transmission"] = 0.5
# cust_params["infectious_period"] = 5
# cust_params["immune_period"] = 365
# cust_params["susc_mask_efficacy"] = 0.3
# cust_params["inf_mask_efficacy"] = 0.5
# cust_params["nomask_social"] = 1
# cust_params["nomask_fear"] = 0.5
# cust_params["mask_social"] = 1
# cust_params["mask_fear"] = 10

pars.update(cust_params)

# Model parameters  # todo: I highly recommend starting a parameter file (csv, json, whatever data format) ASAP, and noting symbol, description, expected value, min, max values, and any sources. You can then read them into python and generate tex tables from single source to change.
beta = pars.get("transmission")  # Infectiousness of disease
c = pars.get("susc_mask_efficacy")  # Effectiveness of susc. wearing mask
p = pars.get("inf_mask_efficacy")  # Effectiveness of infected wearing mask
gamma = pars.get("infectious_period")  # Recovery rate
nu = pars.get("immune_period")   # Immunity
mu = 1/pars.get("av_lifespan")   # av life ~70 years

if gamma != 0:
    gamma = 1/gamma
if nu != 0:
    nu = 1/nu

a1 = pars.get("nomask_social")  # Social influence on mask wearers
a2 = pars.get("nomask_fear")  # Fear of disease for mask wearers
w1 = pars.get("mask_social")   # Social influence on non-mask wearers
w2 = pars.get("mask_fear")   # Fear of disease for non-mask wearers


# Time steps/number of days
TS = 1.0
ND = 50.0

t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end+t_inc, t_inc)

# Inital parameters
S0_m = 0.
I0_m = 0
I0_n = 1e-2  # 10% of the population starting infectious is quite high
R0_n = 0
R0_m = 0
S0_n = 1 - S0_m - I0_m - I0_n - R0_n - R0_m
INPUT = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n)

# %% Differential equations set up

# Change actioned


def rate_to_infect(Im, In):
    return beta * (In + (1 - p) * Im)


# todo: it *might* be faster to pre-add then pass things e.g. on line 95 `rate_to_mask(tot_mask_pop=V[0]+V[2]+V[4], tot_inf=V[2]+V[3])` and corresponding changes here
# Change actioned
def rate_to_mask(tot_mask_prop, tot_inf):
    return w1 * (tot_mask_prop) + w2 * (tot_inf)


# todo: if make change suggested on line 74, do 1-summed terms to pass to here, call them `total_no_mask_pop` and `tot_uninf` or similar
# Change actioned
def rate_to_no_mask(tot_no_mask_prop, tot_uninf):
    return a1 * (tot_no_mask_prop) + a2 * (tot_uninf)


# Added demography, simple birth/death
def diff_eqs(t, INP):
    '''The main set of equations.
    :params INP: previous population values (list of 6 values)
    :params t: current time (scalar)
    '''
    Y = np.zeros((len(INP)))
    V = INP

    tot_mask_prop = V[0] + V[2] + V[4]
    tot_inf = V[2] + V[3]

    lam = rate_to_infect(V[2], V[3])
    omega = rate_to_mask(tot_mask_prop=tot_mask_prop, tot_inf=tot_inf)
    alpha = rate_to_no_mask(tot_no_mask_prop=1 -
                            tot_mask_prop, tot_uninf=1 - tot_inf)

    Y[0] = -lam * (1 - c) * V[0] - alpha * V[0] + \
        omega * V[1] + nu * V[4] + mu - mu*V[0]  # S_m
    Y[1] = -lam * V[1] + alpha * V[0] - \
        omega * V[1] + nu * V[5] - mu*V[1]  # S_n
    Y[2] = lam * (1 - c) * V[0] - alpha * V[2] + \
        omega * V[3] - gamma * V[2] - mu*V[2]  # I_m
    Y[3] = lam * V[1] + alpha * V[2] - omega * \
        V[3] - gamma * V[3] - mu*V[3]  # I_n
    Y[4] = gamma * (V[2]) - nu * V[4] - alpha * V[4] + \
        omega * V[5] - mu*V[4]  # R_m
    Y[5] = gamma * (V[3]) - nu * V[5] + alpha * V[4] - \
        omega * V[5] - mu*V[5]  # R_n
    return Y   # For odeint

# %%


# todo: why not change `INP` to `V`?
# Changed to Y to distinguish from V matrix
# def ngm_reproduction_number(beta, c, p, a1, a2, w1, w2, gamma, Y):

#     Im = Y[2]
#     In = Y[3]

#     tot_mask_prop = Y[0] + Y[2] + Y[4]
#     tot_inf = Y[2] + Y[3]

#     omega = rate_to_mask(tot_mask_prop=tot_mask_prop, tot_inf=tot_inf)
#     alpha = rate_to_no_mask(tot_no_mask_prop=1 -
#                             tot_mask_prop, tot_uninf=1 - tot_inf)

#     F = np.array(
#         [[beta * (1-c) * (1 - p) * Y[0], beta * (1-c) * Y[0]], [beta * (1 - p) * Y[1], beta * Y[1]]])

#     V = np.array([[alpha + gamma - a2 * Im - (w1 + w2) * In, (a1 - a2) * Im - omega - w2 * In],
#                   [(w1 + w2) * In - alpha + a2 * Im, omega + gamma - (a1 - a2) * Im + w2 * In]])
#     Vinv = np.linalg.inv(V)

#     prod = np.matmul(F, Vinv)

#     ANS = np.linalg.eigvals(prod)

#     return np.absolute(ANS).max()


def ngm_reproduction_number(beta, c, p, a1, a2, w1, w2, gamma, Y):

    Im = Y[2]
    In = Y[3]

    tot_mask_prop = Y[0] + Y[2] + Y[4]
    tot_inf = Y[2] + Y[3]

    omega = rate_to_mask(tot_mask_prop=tot_mask_prop, tot_inf=tot_inf)
    alpha = rate_to_no_mask(tot_no_mask_prop=1 -
                            tot_mask_prop, tot_uninf=1 - tot_inf)

    x = alpha - a2 * Im - (w1 + w2) * In
    y = -(a1 - a2) * Im + omega + w2 * In

    a = (1 - p) * (gamma + mu + y) + x
    b = (1 - p) * y + gamma + mu + x

    Gamma = beta/((gamma + mu) * ((gamma + mu) + x + y))

    return Gamma * ((1-c) * Y[0] * a + Y[1] * b)


def cal_rt(t):
    ANS = ngm_reproduction_number(
        beta, c, p, a1, a2, w1, w2, gamma, dat[t, :])
    return ANS

# %% Run ODE-int

# Change actioned
# fixme: the documentation says to use spi.solve_ivp for new code, noting opposite orders of expected function inputs etc
# RES = spi.odeint(diff_eqs, INPUT, t_range)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html


RES = spi.solve_ivp(diff_eqs, [t_start, t_end],  INPUT, t_eval=t_range)
dat = RES.y.T
# t_range = RES.t

Rt = list(map(cal_rt, range(len(t_range))))

switch_time = next(i for i, V in enumerate(Rt) if V <= 1)
tt = t_range[switch_time]

plt.figure()
plt.plot(t_range, Rt)
plt.plot([t_range[0], t_range[-1]], [1, 1], ':k')
plt.plot([tt, tt], [0, 2], ':k')
plt.xlabel("time")
plt.ylabel("R_t")
plt.show()

# %% plotting

# Everything everywhere all at once
plt.figure()
plt.plot(dat[:, 0], label="Susceptibles - Mask")
plt.plot(dat[:, 1], label="Susceptibles - No Mask")
plt.plot(dat[:, 2], label="Infectious - Mask")
plt.plot(dat[:, 3], label="Infectious - No Mask")
plt.plot(dat[:, 4], label="Recovereds - Mask")
plt.plot(dat[:, 5], label="Recovereds - No Mask")
plt.legend()
plt.xlabel("time")
plt.ylabel("proportion")
plt.show()

# S I R
plt.figure()
plt.plot(dat[:, 0] + dat[:, 1], color="y", label="Susceptibles")
plt.plot(dat[:, 2] + dat[:, 3], color="g", label="Infectious")
plt.plot(dat[:, 4] + dat[:, 5], color="r", label="Recovereds")
plt.plot([tt, tt], [0, 1], ':k')
plt.legend()
plt.xlabel("time")
plt.ylabel("proportion")
plt.show()

# Masks
plt.figure()
plt.plot(dat[:, 0] + dat[:, 2] + dat[:, 4], label="Masks")
plt.plot(dat[:, 1] + dat[:, 3] + dat[:, 5], label="No Masks")
plt.legend()
plt.xlabel("time")
plt.ylabel("proportion")
plt.show()

# Sub plots
fig, axs = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle('Masked compartments')
axs[0].set_title("Masks")
axs[0].plot(t_range, dat[:, 0], color="y", label="Susceptibles")
axs[0].plot(t_range, dat[:, 2], color="g", label="Infectious")
axs[0].plot(t_range, dat[:, 4], color="r", label="Recovereds")
axs[0].legend()

axs[1].set_title("No Masks")
axs[1].plot(t_range, dat[:, 1], color="y", label="Susceptibles")
axs[1].plot(t_range, dat[:, 3], color="g", label="Infectious")
axs[1].plot(t_range, dat[:, 5], color="r", label="Recovereds")
axs[1].legend()
fig.tight_layout()
plt.show()
