#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:30:20 2023

@author: rya200
"""

# %% libraries

import numpy as np
import matplotlib.pyplot as plt

# %% Inital parameters


# %% Define events


class sir(object):

    def __init__(self, **kwargs):
        """
        P has shape (Sn, Sb, In, Ib, Rn, Rb), note this is opposite to code in msir_4

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        args = {"beta": 2,
                "gamma": 1,
                "nu": 1/240,
                "a1": 0.5,  # No mask social
                "a2": 0,  # No mask fear of disease
                "a3": 0.01,  # No mask exhaust
                "w1": 0.05 * 8,  # Mask social
                "w2": 8,  # Mask fear of disease
                "w3": 0.01,  # Mask const take up
                "mask_p": 0.8,  # mask effictiveness for infected
                "mask_c": 0.8,  # Mask effectiveness for susc
                "P": np.array([1e4, 1, 1, 0, 0, 0]),
                "t": 0}

        args.update(kwargs)

        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

        self.event_list = ["infect_sn",
                           "infect_sb",
                           "recover_in",
                           "recover_ib",
                           "immune_rn",
                           "immune_rb",
                           "sn_to_sb",
                           "sb_to_sn",
                           "in_to_ib",
                           "ib_to_in",
                           "rn_to_rb",
                           "rb_to_rn"
                           ]
        self.events = [self.event_infect_sn,
                       self.event_infect_sb,
                       self.event_recover_in,
                       self.event_recover_ib,
                       self.event_waning_immunity_rn,
                       self.event_waning_immunity_rb,
                       self.event_gain_mask_sb,
                       self.event_lose_mask_sn,
                       self.event_gain_mask_ib,
                       self.event_lose_mask_in,
                       self.event_gain_mask_rb,
                       self.event_lose_mask_rn
                       ]

    def event_infect_sn(self):
        if self.P[0] > 0:
            self.P[0] -= 1
            self.P[2] += 1

    def event_infect_sb(self):
        if self.P[1] > 0:
            self.P[1] -= 1
            self.P[3] += 1

    def event_recover_in(self):
        if self.P[2] > 0:
            self.P[2] -= 1
            self.P[4] += 1

    def event_recover_ib(self):
        if self.P[3] > 0:
            self.P[3] -= 1
            self.P[5] += 1

    def event_waning_immunity_rn(self):
        if self.P[4] > 0:
            self.P[4] -= 1
            self.P[0] += 1

    def event_waning_immunity_rb(self):
        if self.P[5] > 0:
            self.P[5] -= 1
            self.P[1] += 1

    def event_gain_mask_sb(self):
        if self.P[0] > 0:
            self.P[1] += 1
            self.P[0] -= 1

    def event_gain_mask_ib(self):
        if self.P[2] > 0:
            self.P[3] += 1
            self.P[2] -= 1

    def event_gain_mask_rb(self):
        if self.P[4] > 0:
            self.P[5] += 1
            self.P[4] -= 1

    def event_lose_mask_sn(self):
        if self.P[1] > 0:
            self.P[1] -= 1
            self.P[0] += 1

    def event_lose_mask_in(self):
        if self.P[3] > 0:
            self.P[3] -= 1
            self.P[2] += 1

    def event_lose_mask_rn(self):
        if self.P[5] > 0:
            self.P[5] -= 1
            self.P[4] += 1

    def rates(self):
        N = self.P.sum()

        B = self.P[1:6:2].sum()
        I = self.P[2:4].sum()

        alpha = self.a1 * (N-B)/N + self.a2*(N-I)/N + self.a3
        omega = self.w1 * B/N + self.w2 * I/N + self.w3

        rate_infect_sn = self.beta/N * \
            (self.P[2] + (1-self.mask_p) * self.P[3]) * self.P[0]
        rate_infect_sb = self.beta/N * \
            (self.P[2] + (1-self.mask_p) * self.P[3]) * \
            (1-self.mask_c) * self.P[1]
        rate_recover_in = self.gamma * self.P[2]
        rate_recover_ib = self.gamma * self.P[3]
        rate_immune_rn = self.nu * self.P[4]
        rate_immune_rb = self.nu * self.P[5]
        rate_sn_to_sb = omega*self.P[0]
        rate_sb_to_sn = alpha*self.P[1]
        rate_in_to_ib = omega*self.P[2]
        rate_ib_to_in = alpha*self.P[3]
        rate_rn_to_rb = omega*self.P[4]
        rate_rb_to_rn = alpha*self.P[5]

        return np.array([rate_infect_sn, rate_infect_sb,
                         rate_recover_in, rate_recover_ib,
                         rate_immune_rn, rate_immune_rb,
                         rate_sn_to_sb, rate_sb_to_sn,
                         rate_in_to_ib, rate_ib_to_in,
                         rate_rn_to_rb, rate_rb_to_rn
                         ])

    def perform_event(self):

        rates = self.rates()

        rate_total = rates.sum()

        R1 = np.random.rand()
        R2 = np.random.rand()

        dt = -np.log(R1)/(rate_total)

        p = R2*rate_total

        cum_rates = np.cumsum(rates)

        p_event = [y for y in range(cum_rates.size) if p <= cum_rates[y]][0]

        self.events[p_event]()

        self.t += dt

        return self.event_list[p_event]


def run_iterations(model):
    T = [0]
    res = model.P.reshape(1, 6)
    count = 0

    run_mod = model

    num_inf = 0

    while (T[count] < ND) and (res[-1, 2:4].sum() > 0):
        count += 1
        ev = run_mod.perform_event()
        if count % 50000 == 0:
            print(ev)
        if "infect" in ev:
            num_inf += 1

        T.append(run_mod.t)

        res = np.row_stack([res, run_mod.P])

        # if count > 50000:
        #     break

    return res, T, num_inf

# %%


if __name__ == "__main__":

    ND = 300.0

    num_runs = 1
    plt.figure()
    for x in range(num_runs):

        PP = np.array([1e4, 1, 1, 0, 0, 0])

        R0 = 5
        gamma = 0.4

        args = {"beta": R0*gamma, "gamma": gamma, "P": PP, "t": 0,
                "mask_c": 0.8, "mask_p": 0.8
                }

        mod = sir(**args)

        res, t, num_inf = run_iterations(mod)

# %%

        plt.plot(t, res[:, 0], "y", label="S", alpha=0.6)
        plt.plot(t, res[:, 1], "y:", label="S", alpha=0.6)
        plt.plot(t, res[:, 2], "g", label="I", alpha=0.6)
        plt.plot(t, res[:, 3], "g:", label="I", alpha=0.6)
        plt.plot(t, res[:, 4], "r", label="R", alpha=0.6)
        plt.plot(t, res[:, 5], "r:", label="R", alpha=0.6)
        # plt.legend()
    plt.show()
