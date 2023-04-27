#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:23:59 2023

This file runs the same simulations as in 01_sir_with_masks but tries to formalise it a bit more.
Specifically, creates  SIR object with masks, makes for simplier code running.
Also incolude a main arguement so I can import this into other python scripts?

Looking at adding in samples size N, not just scaling down
Need to include test to see if epidemic is finished

@author: rya200
"""

# %% Packages/libraries
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
        N = PP.sum() - PP[6]

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
        ans = y[2] + y[3] - 1
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
    TS = 1.0
    ND = 600.0

    t_start = 0.0
    t_end = ND
    t_inc = TS
    t_range = np.arange(t_start, t_end+t_inc, t_inc)

    # Inital conditions
    N = 1e6
    # Note the order of conditions (M-N)
    S0_m = 1.
    I0_m = 0
    I0_n = 1  # 1% start infected
    R0_n = 0
    R0_m = 0
    S0_n = N - S0_m - I0_m - I0_n - R0_n - R0_m
    FS = 0
    init_cond = (S0_m, S0_n, I0_m, I0_n, R0_m, R0_n, FS)

    w1 = 8
    R0 = 5
    # Enter custom params
    cust_params = dict()
    cust_params["transmission"] = R0*0.4
    cust_params["infectious_period"] = 1/0.4
    # cust_params["immune_period"] = 240
    cust_params["av_lifespan"] = 0
    cust_params["susc_mask_efficacy"] = 0.8
    cust_params["inf_mask_efficacy"] = 0.8
    cust_params["nomask_social"] = 0.5
    cust_params["nomask_fear"] = 0.
    cust_params["mask_social"] = 0.05 * w1
    cust_params["mask_fear"] = w1
    cust_params["mask_const"] = 0.
    cust_params["nomask_const"] = 0.
    model = msir(**cust_params)

    # Run integrator, convert results to long format
    RES = spi.integrate.solve_ivp(fun=model.run,
                                  t_span=[t_start, t_end],
                                  y0=init_cond,
                                  t_eval=t_range,
                                  events=[event])
    dat = RES.y.T

    toc = time.time()

    print("Script time is %f" % (toc - tic))

    # Rt = list(map(lambda t: model.NGM(dat[t, :]), range(len(t_range))))

    # switch_time = next(i for i, V in enumerate(Rt) if V <= 1)
    # tt = t_range[switch_time]

    # %% plotting

    # sir_R0 = model.transmission / (1/model.infectious_period)
    # sirmn_R0 = model.transmission*(
    #     init_cond[1] + (1 - model.susc_mask_efficacy) * init_cond[0]) / (1/model.infectious_period)

    # plt.figure()
    # plt.plot(t_range, Rt)
    # plt.plot([t_range[0], t_range[-1]], [1, 1], ':k')
    # plt.plot([tt, tt], [0, 2], ':k')
    # plt.plot([t_range[0], t_range[-1]],
    #          [sir_R0,
    #           sir_R0],
    #          ":r",
    #          label="SIR R0")
    # # plt.plot([t_range[0], t_range[-1]],
    # #          [sirmn_R0,
    # #           sirmn_R0],
    # #          ":b",
    # #          label="SIR-MN R0")
    # plt.legend()
    # plt.xlabel("time")
    # plt.ylabel("R_t")
    # plt.show()

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
    # plt.plot([tt, tt], [0, 1], ':k')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("proportion")
    plt.show()

    if model.mask_social - model.nomask_social == 0:
        Delta = (model.nomask_fear + model.nomask_const) / \
            (model.mask_const + model.nomask_fear + model.nomask_const)
    else:
        Delta = ((model.mask_social - model.nomask_social + model.mask_const + model.nomask_fear + model.nomask_const) - np.sqrt((model.mask_social - model.nomask_social + model.mask_const +
                                                                                                                                  model.nomask_fear + model.nomask_const)**2 - 4 * (model.mask_social - model.nomask_social) * (model.nomask_fear + model.nomask_const))) / (2 * (model.mask_social - model.nomask_social))

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
