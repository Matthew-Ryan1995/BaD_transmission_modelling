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

ND = 360.0

# %% Define events


class sir(object):

    def __init__(self, **kwargs):
        args = {"beta": 2, "gamma": 1, "P": np.array([1e4, 1, 0]), "t": 0}

        args.update(kwargs)

        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

    def event_infect(self):
        self.P[0] -= 1
        self.P[1] += 1

    def event_recover(self):
        self.P[1] -= 1
        self.P[2] += 1

    def rates(self):
        N = self.P.sum()

        rate_infect = self.beta * self.P[0] * self.P[1]/N
        rate_recover = self.gamma * self.P[1]

        return np.array([rate_infect, rate_recover])

    def perform_event(self):

        rates = self.rates()

        rate_total = rates.sum()

        R1 = np.random.rand()
        R2 = np.random.rand()

        dt = -np.log(R1)/(rate_total)

        p = R2*rate_total

        if p <= rates[0]:
            self.event_infect()
        else:
            self.event_recover()

        self.t += dt


def run_iterations(model):
    T = [0]
    res = model.P.reshape(1, 3)
    count = 0

    run_mod = model

    while (T[count] < ND) and (res[-1, 1] > 0):
        count += 1
        run_mod.perform_event()

        T.append(run_mod.t)

        res = np.row_stack([res, run_mod.P])

    return res, T

# %%


if __name__ == "__main__":

    plt.figure()
    for x in range(10):

        PP = np.array([1e4, 1, 0])

        R0 = 2
        gamma = 0.2

        args = {"beta": R0*gamma, "gamma": gamma, "P": PP, "t": 0}

        mod = sir(**args)

        res, t = run_iterations(mod)

# %%

        plt.plot(t, res[:, 0], "y", label="S", alpha=0.6)
        plt.plot(t, res[:, 1], "g", label="I", alpha=0.6)
        plt.plot(t, res[:, 2], "r", label="R", alpha=0.6)
        # plt.legend()
    plt.show()
