#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:23:59 2023

This file runs the same simulations as in 01_sir_with_masks but tries to formalise it a bit more.
Specifically, creates  SIR object with masks, makes for simplier code running.
Also incolude a main arguement so I can import this into other python scripts?

Looking at adding in samples size N, not just scaling down
Need to include test to see if epidemic is finished

Need to look at my equilibirum a little bit more, I can get a switch.  I am still getting the correct equilibirum, but predicting the wrong 
behaviour

@author: rya200
"""

# %% Packages/libraries
import math
from scipy.integrate import quad
from scipy.optimize import fsolve
import numpy as np
import scipy as spi
import matplotlib.pyplot as plt
import json
import os
import time

working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %% Class definitions


class msir(object):
    """
    Implementation of the SIR model with mask states for each compartment.  Explicitly, we assume proportion and not counts.  
    Currently assuming no demograhpy, no death due to pathogen, homogenous mixing, transitions between mask/no mask determined by
    social influence and fear of disease.  Currently assuming FD-like "infection" process for masks with fear of disease.
    """

    def __init__(self, **kwargs):
        """
        Written by: Rosyln Hickson
        Required parameters when initialising this class, plus deaths and births optional.
        :param transmission: double, the transmission rate from those infectious to those susceptible.
        :param infectious_period: scalar, the average infectious period.
        :param immune_period: scalar, average immunity period (for SIRS)
        :param susc_mask_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
        :param inf_mask_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
        :param nomask_social: double, social influence of non-mask wearers on mask wearers (a1)
        :param nomask_fear: double, Fear of disease for mask wearers to remove mask (a2)
        :param mask_social: double, social influence of mask wearers on non-mask wearers (w1)
        :param mask_fear: double, Fear of disease for non-mask wearers to put on mask (w2)
        :param av_lifespan: scalar, average life span in years
        """
        args = self.set_defaults()  # load default values from json file
        # python will overwrite existing values in the `default` dict with user specified values from kwargs
        args.update(kwargs)

        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

    def set_defaults(self, filename="data/parameter_ranges.json"):
        """
        Written by: Rosyln Hickson
        Pull out default values from a file in json format.
        :param filename: json file containing default parameter values, which can be overridden by user specified values
        :return: loaded expected parameter values
        """
        with open(filename) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            json_data[key] = value["exp"]
        return json_data

    def update_params(self, **kwargs):
        args = kwargs
        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

    def rate_to_infect(self, Im, In):
        return self.transmission * (In + (1 - self.inf_mask_efficacy) * Im)

    def rate_to_mask(self, tot_mask_prop, tot_inf):
        return self.mask_social * (tot_mask_prop) + self.mask_fear * (tot_inf) + self.mask_const

    def rate_to_no_mask(self, tot_no_mask_prop, tot_uninf):
        return self.nomask_social * (tot_no_mask_prop) + self.nomask_fear * (tot_uninf) + self.nomask_const

    def run(self, t, PP):
        """
        ODE set up to use spi.integrate.solve_ivp.  This defines the change in state at time t.

        Parameters
        ----------
        t : double
            time point.
        PP : array
            State of the population at time t-1, in proportions.

        Returns
        -------
        Y : array
            rate of change of population compartments at time t.
        """
        Y = np.zeros((len(PP)))
        N = PP.sum() - PP[6] - PP[7]

        tot_mask_prop = (PP[0] + PP[2] + PP[4])/N
        tot_inf = (PP[2] + PP[3])/N

        lam = self.rate_to_infect(PP[2]/N, PP[3]/N)
        omega = self.rate_to_mask(tot_mask_prop=tot_mask_prop,
                                  tot_inf=tot_inf)
        alpha = self.rate_to_no_mask(tot_no_mask_prop=1-tot_mask_prop,
                                     tot_uninf=1-tot_inf)

        if self.immune_period == 0:
            nu = 0
        else:
            nu = 1/self.immune_period
        if self.infectious_period == 0:
            gamma = 0
        else:
            gamma = 1/self.infectious_period
        if self.av_lifespan == 0:
            mu = 0
        else:
            mu = 1/self.av_lifespan

        # todo: clean up these equations
        Y[0] = -lam * (1 - self.susc_mask_efficacy) * PP[0] - alpha * PP[0] + \
            omega * PP[1] + nu * PP[4] + mu - mu * PP[0]  # S_m
        Y[1] = -lam * PP[1] + alpha * PP[0] - omega * \
            PP[1] + nu * PP[5] - mu * PP[1]  # S_n
        Y[2] = lam * (1 - self.susc_mask_efficacy) * PP[0] - alpha * PP[2] + \
            omega * PP[3] - gamma * PP[2] - mu * PP[2]  # I_m
        Y[3] = lam * PP[1] + alpha * PP[2] - \
            omega * PP[3] - gamma * PP[3] - mu * PP[3]  # I_n
        Y[4] = gamma * (PP[2]) - nu * PP[4] - alpha * \
            PP[4] + omega * PP[5] - mu * PP[4]  # R_m
        Y[5] = gamma * (PP[3]) - nu * PP[5] + alpha * \
            PP[4] - omega * PP[5] - mu * PP[5]  # R_n
        Y[6] = lam * (1 - self.susc_mask_efficacy) * \
            PP[0] + lam * PP[1]  # Final Size
        Y[7] = lam * (1 - self.susc_mask_efficacy) * \
            PP[0]  # Final Size for masks

        # assert np.isclose(Y[0:6].sum(), 0)
        return Y

    def NGM(self, CP):
        """


        Parameters
        ----------
        CP : array
            Current population proportions.

        Returns
        -------
        Double
            Largest (absolute) eigenvalue for NGM.

        """
        N = CP.sum()
        Im = CP[2] / N
        In = CP[3] / N

        if self.infectious_period == 0:
            gamma = 0
        else:
            gamma = 1/self.infectious_period
        if self.av_lifespan == 0:
            mu = 0
        else:
            mu = 1/self.av_lifespan

        gamma = gamma + mu

        tot_mask_prop = (CP[0] + CP[2] + CP[4]) / N
        tot_inf = (CP[2] + CP[3]) / N

        omega = self.rate_to_mask(tot_mask_prop=tot_mask_prop, tot_inf=tot_inf)
        alpha = self.rate_to_no_mask(tot_no_mask_prop=1 -
                                     tot_mask_prop, tot_uninf=1 - tot_inf)

        x = alpha - self.nomask_fear * Im - \
            (self.mask_social + self.mask_fear) * In
        y = -(self.nomask_social - self.nomask_fear) * \
            Im + omega + self.mask_fear * In

        a = (1 - self.inf_mask_efficacy) * (gamma + y) + x
        b = (1 - self.inf_mask_efficacy) * y + gamma + x

        Gamma = self.transmission/(gamma * (gamma + x + y))

        return Gamma * ((1-self.susc_mask_efficacy) * CP[0] * a + CP[1] * b) / N


def event(t, y):
    if t > 10:
        ans = y[2] + y[3] - 1e-3
    else:
        ans = 1
    return ans


event.terminal = True

if __name__ == "__main__":
    """
        Run the SIR with masks model, display outputs
    """

    tic = time.time()
    # Time steps/number of days for the disease
    TS = 1.0/14
    ND = 600.0

    t_start = 0.0
    t_end = ND
    t_inc = TS
    t_range = np.arange(t_start, t_end+t_inc, t_inc)

    # Inital conditions
    N = 1
    # Note the order of conditions (M-N)
    S0_m = 1e-6
    I0_m = 0
    I0_n = 1e-3  # 1% start infected
    R0_n = 0
    R0_m = 0
    S0_n = N - S0_m - I0_m - I0_n - R0_n - R0_m
    FS = 0
    FS_m = 0
    init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n, FS, FS_m)

    w1 = 8
    R0 = 5
    gamma = 0.4
    # Enter custom params
    # cust_params = dict()
    # cust_params["transmission"] = R0*0.1
    # cust_params["infectious_period"] = 1/0.1
    # # cust_params["immune_period"] = 240
    # cust_params["av_lifespan"] = 0
    # cust_params["susc_mask_efficacy"] = 0.4
    # cust_params["inf_mask_efficacy"] = 0.8
    # cust_params["nomask_social"] = 0.
    # cust_params["nomask_fear"] = 0.0
    # cust_params["mask_social"] = w1  # 0.0 * w1
    # cust_params["mask_fear"] = 0  # w1
    # cust_params["mask_const"] = 0.
    # cust_params["nomask_const"] = 0.01
    # model = msir(**cust_params)
    cust_params = dict()
    cust_params["transmission"] = R0*gamma
    cust_params["infectious_period"] = 1/gamma
    cust_params["immune_period"] = 240
    cust_params["av_lifespan"] = 0
    cust_params["susc_mask_efficacy"] = 0.8
    cust_params["inf_mask_efficacy"] = 0.4
    cust_params["nomask_social"] = 0.5
    cust_params["nomask_fear"] = 1e-1
    cust_params["mask_social"] = 0.05 * w1
    cust_params["mask_fear"] = w1
    cust_params["mask_const"] = 0.01
    cust_params["nomask_const"] = 0.01
    model = msir(**cust_params)

    R0_b = (model.mask_social) / \
        (model.nomask_fear + model.nomask_const +
         model.nomask_social) + model.mask_fear/(1/model.infectious_period)

    # S0_m = 1-Delta
    # I0_m = 0
    # I0_n = 1e-6  # 1% start infected
    # R0_n = 0
    # R0_m = 0
    # S0_n = Delta - I0_n
    # FS = 0
    # FS_m = 0
    # init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n, FS, FS_m)

    # Run integrator, convert results to long format
    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=init_cond,
                                  t_eval=t_range,
                                  # events=[event]
                                  )
    dat = RES.y.T

    toc = time.time()

    # model.rate_to_mask(dat[-1, 0:5:2].sum(), dat[-1, 2:4].sum())
    # model.rate_to_no_mask(1-dat[-1, 0:5:2].sum(), 1-dat[-1, 2:4].sum())

    print("Script time is %f" % (toc - tic))

    Rt = list(map(lambda t: model.NGM(dat[t, :]), range(dat.shape[0])))

    # switch_time = next(i for i, V in enumerate(Rt) if V <= 1)
    switch_time = 0
    tt = t_range[switch_time]

    if model.mask_social - model.nomask_social != 0:
        Delta = ((model.mask_social - model.nomask_social + model.mask_const + model.nomask_fear + model.nomask_const) - np.sqrt((model.mask_social - model.nomask_social + model.mask_const +
                                                                                                                                  model.nomask_fear + model.nomask_const)**2 - 4 * (model.mask_social - model.nomask_social) * (model.nomask_fear + model.nomask_const))) / (2 * (model.mask_social - model.nomask_social))
    elif (model.mask_const + model.nomask_fear + model.nomask_const) == 0:
        Delta = S0_m/N
    else:
        Delta = (model.nomask_fear + model.nomask_const) / \
            (model.mask_const + model.nomask_fear + model.nomask_const)

    # %% plotting

    sir_R0 = model.transmission / (1/model.infectious_period)
    sirmn_R0 = model.transmission*(
        init_cond[1] + (1 - model.susc_mask_efficacy) * init_cond[0]) / (1/model.infectious_period)

    plt.figure()
    plt.plot(Rt)
    plt.plot([t_range[0], t_range[-1]], [1, 1], ':k')
    plt.plot([tt, tt], [0, 2], ':k')
    plt.plot([t_range[0], t_range[-1]],
             [sir_R0,
              sir_R0],
             ":r",
             label="SIR R0")
    # plt.plot(R0 * (dat[:, 1]/N + (1-0.8) * dat[:, 0]/N)) # Almost R_eff in a much simpler argument
    # plt.plot([t_range[0], t_range[-1]],
    #          [sirmn_R0,
    #           sirmn_R0],
    #          ":b",
    #          label="SIR-MN R0")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("R_t")
    plt.show()

    # Everything everywhere all at once
    # K = (model.mask_const + 1/model.infectious_period - (model.nomask_const**2)/model.mask_const) / \
    #     (model.transmission * (1 + (1 - model.inf_mask_efficacy)
    #      * model.nomask_const / model.mask_const))
    # K2 = model.mask_const / model.nomask_const * K
    plt.figure()
    plt.plot(dat[:, 0], label="Susceptibles - Mask")
    plt.plot(dat[:, 1], label="Susceptibles - No Mask")
    # plt.plot([0, dat.shape[0]], [K, K], ":k")
    # plt.plot([0, dat.shape[0]], [K2, K2], ":k")
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
    plt.plot([15, 15], [0, N], ':k')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("proportion")
    plt.show()

    # Masks
    plt.figure()
    plt.plot((dat[:, 0] + dat[:, 2] + dat[:, 4])/N, label="Masks")
    plt.plot((dat[:, 1] + dat[:, 3] + dat[:, 5])/N, label="No Masks")

    plt.plot([0, dat[:, 1].size], [Delta, Delta], ":k")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("proportion")
    plt.show()

    # plt.figure(figsize=[8, 8*0.618], dpi=600)
    # plt.title("Masks vs infections")
    # plt.plot(np.sum(dat[:, 2:4], axis=1)/N, np.sum(dat[:, 0:5:2]/N, axis=1))
    # plt.xlabel("Proportion of infectious")
    # plt.ylabel("Proportion of masks")
    # plt.savefig(fname="img/mask_vs_infected.png")
    # plt.show()

    # plt.figure(figsize=[8, 8*0.618], dpi=600)
    # plt.title("S vs infections")
    # plt.plot(np.sum(dat[:, 2:4], axis=1)/N, np.sum(dat[:, 0:2], axis=1)/N)
    # plt.xlabel("Proportion of infectious")
    # plt.ylabel("Proportion of S")
    # plt.savefig(fname="img/S_vs_infected.png")
    # plt.show()

    # # Sub plots
    # fig, axs = plt.subplots(2, sharex=True, sharey=True)
    # fig.suptitle('Masked compartments')
    # axs[0].set_title("Masks")
    # axs[0].plot(dat[:, 0], color="y", label="Susceptibles")
    # axs[0].plot(dat[:, 2], color="g", label="Infectious")
    # axs[0].plot(dat[:, 4], color="r", label="Recovereds")
    # axs[0].legend()

    # axs[1].set_title("No Masks")
    # axs[1].plot(dat[:, 1], color="y", label="Susceptibles")
    # axs[1].plot(dat[:, 3], color="g", label="Infectious")
    # axs[1].plot(dat[:, 5], color="r", label="Recovereds")
    # axs[1].legend()
    # fig.tight_layout()
    # plt.show()

    # # Infection
    # a = 0
    # plt.figure()
    # # plt.plot(dat[:, 2], color="green", linestyle="dashed", label="I - Mask")
    # # plt.plot(dat[:, 3], color="green", linestyle=":", label="I - No Mask")
    # plt.plot(dat[a:, 2] + dat[a:, 3], color="green", label="I")
    # plt.legend()
    # plt.show()

    # tmp = spi.signal.find_peaks(
    #     dat[:, 2] + dat[:, 3], height=(10, None), prominence=(10, None))

    # fig, ax = plt.subplots()
    # ax.set_title("All states")
    # ax.plot(dat[:, 0] + dat[:, 2] + dat[:, 4], label="Masks",
    #         c="y", linestyle="-")
    # ax.plot(dat[:, 1] + dat[:, 3] + dat[:, 5], label="No Masks",
    #         c="b", linestyle="--")

    # ax.set_ylabel("Mask uptake")
    # ax.set_xlabel("Days")
    # plt.legend()
    # ax2 = ax.twinx()
    # ax2.plot(dat[:, 2] + dat[:, 3], color="green", label="I", linestyle=":")
    # ax2.set_ylabel("Infections")
    # ax2.ticklabel_format(style="sci")
    # plt.legend()
    # plt.show()

# %%


# def find_final_size(x):
#     p_N = Delta

#     lam_mn = 1-cust_params["susc_mask_efficacy"]
#     lam_mm = (1-cust_params["susc_mask_efficacy"]) * \
#         (1 - cust_params["inf_mask_efficacy"])

#     lam_mn *= Rt[0]
#     lam_mm *= Rt[0]

#     init = S0_m / N

#     return x - (1 - init * np.exp(-x * (p_N * (lam_mn - lam_mm) + lam_mm)))


# tmp = fsolve(find_final_size, x0=[0])
# tmp2 = dat[-1, 7]/N
# print("Est FS = %f" % (tmp))
# print("True FS = %f" % (tmp2))

# %%

# S_tot = np.sum(dat[:, 2:4], axis=1)/N
# M_tot = np.sum(dat[:, 0:5:2], axis=1)/N
# N_tot = 1-M_tot
# S_M = dat[:, 2]/N
# S_N = dat[:, 3]/N

# ttt = S_tot.size
# plt.figure()
# plt.plot(S_M, S_tot*M_tot)
# plt.show()
# # plt.plot([0, 1], [0, 1])
# # plt.plot(S_M[0:ttt], (S_tot*M_tot)[0:ttt])

# plt.figure()
# plt.plot(S_N + S_M, S_tot)
# plt.show()

# %%

# n1 = 1-Delta
# n2 = Delta


# def FS_eqn(x):
#     y = np.zeros(2)

#     y[0] = 1 - np.exp(-R0 * x[0]) - x[0]
#     y[1] = 1 - np.exp(-R0 * x[1] * (1 - cust_params["susc_mask_efficacy"])
#                       * (1 - cust_params["inf_mask_efficacy"])) - x[1]

#     return y


# est_fs = fsolve(FS_eqn, np.array([0.5, 0.5]))

# # %%

# beta = cust_params["transmission"]
# p = cust_params["inf_mask_efficacy"]
# c = cust_params["susc_mask_efficacy"]
# nu = 1/cust_params["immune_period"]
# gamma = 1/cust_params["infectious_period"]

# i1 = dat[-1, 3]
# D = dat[-1, 1:6:2].sum()
# B = np.array([[beta, (1-p) * beta], [(1-p) * beta, (1-c) * (1-p) * beta]])


# i2 = (nu * gamma - B[0, 0] * (nu * D - (nu + gamma) * i1)
#       ) * i1 / (B[0, 1] * (nu * D - (nu + gamma) * i1))

# s1 = nu * (D - i1) / (B[0, 0] * i1 + B[0, 1] * i2 + nu)
# r1 = D - s1 - i1

# s2 = nu * (1 - D - i2)/(B[1, 0] * i1 + B[1, 1] * i2 + nu)
# r2 = 1 - D - s2 - i2

# print(dat[-1, 0:6])
# print([s2, s1, i2, i1, r2, r1])
# print(np.array([s1, s2, i1, i2, r1, r2]).sum())

# # %%

# a = model.rate_to_mask(1-D, dat[-1, 2:4].sum())
# w = model.rate_to_no_mask(D, 1 - dat[-1, 2:4].sum())

# print(a * D - w * (1-D))
# print(a * dat[-1, 1] - w * dat[-1, 0])
# print(a * dat[-1, 3] - w * dat[-1, 2])
# print(a * dat[-1, 5] - w * dat[-1, 4])

# %%

k3 = model.mask_social - model.nomask_social - \
    model.nomask_fear - model.nomask_const
k2 = model.nomask_fear
k1 = model.mask_fear
k0 = model.mask_const


# def get_B(t):
#     if t == 0:
#         return 0
#     I = np.sum(dat[:, 2:4], 1)

#     I_e_int = I[0:(t+1)] * np.exp(-k3 * t_range[0:(t+1)]) * \
#         TS * np.exp(-k2 * np.cumsum(I[0:(t+1)]*TS))
#     I_e_int = I_e_int.sum()

#     e_int = np.exp(-k3 * t_range[0:(t+1)]) * TS * \
#         np.exp(-k2 * np.cumsum(I[0:(t+1)]*TS))
#     e_int = e_int.sum()

#     numer = k1 * I_e_int + k0 * e_int
#     denom = k3 * e_int + k2 * I_e_int + \
#         np.exp(-k3 * t_range[t+1]) * np.exp(-k2 * np.sum(I[0:(t+1)]*TS))

#     return numer/denom

def sirs_eqn(t, PP):
    Y = np.zeros(3)

    Y[0] = -model.transmission * PP[0] * PP[1] + 1/model.immune_period * PP[2]
    Y[1] = model.transmission * PP[0] * PP[1] - \
        1/model.infectious_period * PP[1]
    Y[2] = 1/model.infectious_period * PP[1] - 1/model.immune_period * PP[2]
    return Y


sirs_res = spi.integrate.solve_ivp(
    sirs_eqn, [t_start, t_end], y0=init_cond[1:6:2], t_eval=t_range)
sirs_dat = sirs_res.y.T


def get_B2(t):
    # if t == 0:
    #     return 0

    # I = np.sum(dat[:, 2:4], 1)
    I = sirs_dat[:, 1]

    integrand = np.cumsum(k3 + k2 * I[0:(t+1)]) * TS

    A1 = np.exp(integrand[-1])

    A2 = np.exp(-integrand) * (k0 + k1 * I[0:(t+1)]) * TS
    A2 = A2.sum()

    return A1 * A2


def get_B(t):
    # if t == 0:
    #     return 0

    I = np.sum(dat[:, 2:4], 1)

    c = k3 + k2*I[0]
    k = k0 + k1*I[0]

    return k/c*(np.exp(c*t_range[t]) - 1)


B = []
B2 = []
ttt = 8000

for t in range(ttt):
    B.append(get_B(t))
for t in range(ttt):
    B2.append(get_B2(t))


a0 = dat[0, 2:4].sum()
a1 = (model.transmission * (1 - a0) - 1/model.infectious_period) * a0
a2 = model.transmission * (1-a0) * a0 * (model.transmission * (1 - 2 * a0) - (model.susc_mask_efficacy +
                                                                              model.inf_mask_efficacy) * (model.mask_fear * a0 + model.mask_const)) - gamma * a1
a2 = a2/2


def get_B_linear(t):

    int_1, err = quad(lambda x: k3 + k2 * (a0 + a1 * x), 0, t)

    def inner_int(zeta):
        int_2, err = quad(lambda x: k3 + k2 * (a0 + a1 * x), 0, zeta)

        return np.exp(-int_2) * (k0 + k1 * (a0 + a1 * zeta))

    int_3, err = quad(inner_int, 0, t)

    return np.exp(int_1) * int_3


def get_B_quad(t):

    int_1, err = quad(lambda x: k3 + k2 * (a0 + a1 * x + a2 * x**2), 0, t)

    def inner_int(zeta):
        int_2, err = quad(lambda x: k3 + k2 *
                          (a0 + a1 * x + a2 * x**2), 0, zeta)

        return np.exp(-int_2) * (k0 + k1 * (a0 + a1 * zeta + a2 * zeta**2))

    int_3, err = quad(inner_int, 0, t)

    return np.exp(int_1) * int_3


def get_B_exp(t):

    int_1, err = quad(lambda x: k3 + k2 * (a0 *
                      np.exp((model.transmission - 1/model.infectious_period) * x)), 0, t)

    def inner_int(zeta):
        int_2, err = quad(lambda x: k3 + k2 *
                          (a0*np.exp((model.transmission - 1/model.infectious_period) * x)), 0, zeta)

        return np.exp(-int_2) * (k0 + k1 * (a0*np.exp((model.transmission - 1/model.infectious_period) * zeta)))

    int_3, err = quad(inner_int, 0, t)

    return np.exp(int_1) * int_3


def get_B_cosh(t):

    def I_est(zeta):
        gamma = 1/model.infectious_period
        a = np.sqrt((S0_n * R0 - 1)**2 + 2*S0_n*I0_n*R0**2)

        phi = math.atanh((S0_n * R0 - 1)/a)

        ANS = a**2 / (2 * S0_n * R0**2) * \
            (math.cosh(a*gamma*zeta/2 - phi))**(-2)

        return ANS

    int_1, err = quad(lambda x: k3 + k2 * I_est(x), 0, t)

    def inner_int(zeta):
        int_2, err = quad(lambda x: k3 + k2 * I_est(x), 0, zeta)

        return np.exp(-int_2) * (k0 + k1 * I_est(zeta))

    int_3, err = quad(inner_int, 0, t)

    return np.exp(int_1) * int_3


B3 = []
B4 = []
B5 = []
B6 = []
for t in t_range[0:ttt]:
    B3.append(get_B_linear(t))
    B4.append(get_B_quad(t))
    # B5.append(get_B_exp(t))
    B6.append(get_B_cosh(t))

# %%

t_first_plot = 45

plt.figure()
plt.title("Estimate of behavioural equation\n B_0 = 10^{-6}, I_0 = 0.001")
plt.plot(t_range[0:t_first_plot], B[0:t_first_plot],
         "b", label="estimate constant")
plt.plot(t_range[0:t_first_plot], B2[0:t_first_plot],
         "g", label="estimate SIRS")
plt.plot(t_range[0:t_first_plot], B3[0:t_first_plot],
         "y", label="estimate linear")
plt.plot(t_range[0:t_first_plot], B4[0:t_first_plot],
         "orange", label="estimate quad")
# plt.plot(t_range[0:t_first_plot], B5[0:t_first_plot],
# "purple", label="estimate exp")
plt.plot(t_range[0:t_first_plot], B6[0:t_first_plot],
         "brown", label="estimate K&R")
plt.plot(t_range[0:t_first_plot], np.sum(
    dat[0:t_first_plot, 0:5:2], 1), "r:", label="truth")
# plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 2:4], 1), "y", label="I")
plt.xlabel("time")
plt.ylabel("Proportion performing behaviour")
plt.legend()
plt.show()

plt.figure()
plt.title(
    "Estimate of behavioural equation for long time\n B_0 = 10^{-6}, I_0 = 0.001")
plt.plot(t_range[0:ttt], B, "b", label="estimate constant")
plt.plot(t_range[0:ttt], B2, "g", label="estimate SIRS")
# plt.plot(t_range[0:ttt], B3[0:ttt], "y", label="estimate linear")
# plt.plot(t_range[0:ttt], B4[0:ttt], "orange", label="estimate quad")
plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 0:5:2], 1), "r", label="truth")
# plt.plot(t_range[0:ttt], B5[0:ttt], "purple", label="estimate exp")
# plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 2:4], 1), "y", label="I")
plt.plot(t_range[0:ttt], B6[0:ttt],
         "brown", label="estimate K&R")
plt.xlabel("time")
plt.ylabel("Proportion performing behaviour")
plt.legend()
plt.show()

plt.figure()
plt.title(
    "Estimate of behavioural equation for super small time\n B_0 = 10^{-6}, I_0 = 0.001")
plt.plot(t_range[0:10], B[0:10], "b", label="estimate constant")
plt.plot(t_range[0:10], B2[0:10], "g", label="estimate SIRS")
plt.plot(t_range[0:10], B3[0:10], "y", label="estimate linear")
plt.plot(t_range[0:10], B4[0:10], "orange", label="estimate quad")
# plt.plot(t_range[0:10], B5[0:10], "purple", label="estimate exp")
plt.plot(t_range[0:10], B6[0:10],
         "brown", label="estimate K&R")
plt.plot(t_range[0:10], np.sum(dat[0:10, 0:5:2], 1), "r:", label="truth")
# plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 2:4], 1), "y", label="I")
plt.xlabel("time")
plt.ylabel("Proportion performing behaviour")
plt.legend()
plt.show()


# %%

# k3 = model.mask_social - model.nomask_social - \
#     model.nomask_fear - model.nomask_const
# k2 = model.nomask_fear
# k1 = model.mask_fear
# k0 = model.mask_const


# def k3k2I(t):
#     i = [i for i in range(len(t_range)) if t_range[i] <= t][-1]
#     i = int(i)
#     I = np.sum(dat[:, 2:4], 1)

#     return k3 + k2 * I[i]


# def k0k1I(t):
#     i = [i for i in range(len(t_range)) if t_range[i] <= t][-1]
#     i = int(i)
#     I = np.sum(dat[:, 2:4], 1)

#     return k0 + k1 * I[i]


# def get_B3(t):
#     t=int(t)
#     integrate_1, err = quad(k3k2I, 0, t)

#     def inner_quad(x):
#         y, err = quad(k3k2I, 0, x)
#         y = np.exp(-y)
#         z = k0k1I(x)

#         return y * z

#     integrate_2, err = quad(inner_quad, 0, t)

#     return np.exp(integrate_1) * integrate_2


# B = []
# B2 = []
# B3 = []
# ttt = 55

# for t in range(ttt):
#     B.append(get_B(t))
# for t in range(ttt):
#     B2.append(get_B2(t))
# for t in range(ttt):
#     B3.append(get_B3(t_range[t]))

# plt.figure()
# plt.title("Estimate of behavioural equation\n B_0 = 10^{-6}, I_0 = 0.001")
# plt.plot(t_range[0:ttt], B, "b", label="Cestimate constant")
# plt.plot(t_range[0:ttt], B2, "g", label="estimate")
# plt.plot(t_range[0:ttt], B3, "g", label="estimate with quad")
# plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 0:5:2], 1), "r", label="truth")
# # plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 2:4], 1), "y", label="I")
# plt.xlabel("time")
# plt.ylabel("Proportion performing behaviour")
# plt.legend()
# plt.show()

# # plt.figure()
# # plt.title(
# #     "Estimate of behavioural equation for super small time\n B_0 = 10^{-6}, I_0 = 0.001")
# # plt.plot(t_range[0:10], B[0:10], "b", label="Cestimate constant")
# # plt.plot(t_range[0:10], B2[0:10], "g", label="estimate")
# # plt.plot(t_range[0:10], np.sum(dat[0:10, 0:5:2], 1), "r", label="truth")
# # # plt.plot(t_range[0:ttt], np.sum(dat[0:ttt, 2:4], 1), "y", label="I")
# # plt.xlabel("time")
# # plt.ylabel("Proportion performing behaviour")
# # plt.legend()
# # plt.show()
