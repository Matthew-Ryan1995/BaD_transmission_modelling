#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:34:08 2023

Trying to foRbalise the BaD model into something a little more usable
If I rewrite the model to get rid of the 1- in the alpha teRbs, I will need to rework this

@author: rya200
"""


# %% Packages/libraries
import math
from scipy.integrate import quad, solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time

working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
os.chdir(working_path)

# %% Class definitions


class bad(object):
    """
    Ibplementation of the SIR model with mask states for each compartment.  Explicitly, we assume proportion and not counts.
    Currently assuming no demograhpy, no death due to pathogen, homogenous mixing, transitions between mask/no mask deteRbined by
    social influence and fear of disease.  Currently assuming FD-like "infection" process for masks with fear of disease.
    """

    def __init__(self, **kwargs):
        """
        Written by: Rosyln Hickson
        Required parameters when initialising this class, plus deaths and births optional.
        :param transmission: double, the transmission rate from those infectious to those susceptible.
        :param infectious_period: scalar, the average infectious period.
        :param immune_period: scalar, average Ibmunity period (for SIRS)
        :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S wears mask (c)
        :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I wears mask (p)
        :param N_social: double, social influence of non-mask wearers on mask wearers (a1)
        :param N_fear: double, Fear of disease for mask wearers to remove mask (a2)
        :param B_social: double, social influence of mask wearers on non-mask wearers (w1)
        :param B_fear: double, Fear of disease for non-mask wearers to put on mask (w2)
        :param av_lifespan: scalar, average life span in years
        """
        args = self.set_defaults()  # load default values from json file
        # python will overwrite existing values in the `default` dict with user specified values from kwargs
        args.update(kwargs)

        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

    def set_defaults(self, filename="data/BaD_parameter_ranges.json"):
        """
        Written by: Rosyln Hickson
        Pull out default values from a file in json foRbat.
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

    def rate_to_infect(self, Ib, In):
        return self.transmission * (In + (1 - self.inf_B_efficacy) * Ib)

    def rate_to_mask(self, tot_B_prop, tot_inf):
        return self.B_social * (tot_B_prop) + self.B_fear * (tot_inf) + self.B_const

    def rate_to_no_mask(self, tot_no_B_prop, tot_uninf):
        return self.N_social * (tot_no_B_prop) + self.N_fear * (tot_uninf) + self.N_const

    def odes(self, t, PP):
        """
        ODE set up to use spi.integrate.solve_ivp.  This defines the change in state at time t.

        Parameters
        ----------
        t : double
            time point.
        PP : array
            State of the population at time t-1, in proportions.
            Assumes that it is of the form:
                [Sn, Sb, In, Ib, Rn, Rb]
                Note that this is reverse from msir_4

        Returns
        -------
        Y : array
            rate of change of population compartments at time t.
        """
        Y = np.zeros((len(PP)))
        P = PP[0:6].sum()

        tot_B_prop = (PP[1:6:2].sum())/P
        tot_inf = (PP[2:4].sum())/P

        lam = self.rate_to_infect(Ib=PP[3]/P, In=PP[2]/P)
        omega = self.rate_to_mask(tot_B_prop=tot_B_prop,
                                  tot_inf=tot_inf)
        alpha = self.rate_to_no_mask(tot_no_B_prop=1-tot_B_prop,
                                     tot_uninf=1-tot_inf)

        # NB: This is slowing things down, if need more spped need to re-evalutate
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

        Y[0] = -lam * PP[0] + alpha * PP[1] - omega * \
            PP[0] + nu * PP[4] - mu * PP[0] + mu * P  # Sn
        Y[1] = -lam * (1-self.susc_B_efficacy) * PP[1] - alpha * \
            PP[1] + omega * PP[0] + nu * PP[5] - mu * PP[1]  # Sb
        Y[2] = lam * PP[0] + alpha * PP[3] - omega * \
            PP[2] - gamma * PP[2] - mu * PP[2]  # In
        Y[3] = lam * (1-self.susc_B_efficacy) * PP[1] - alpha * \
            PP[3] + omega * PP[2] - gamma * PP[3] - mu * PP[3]  # Ib
        Y[4] = gamma * PP[2] + alpha * PP[5] - omega * \
            PP[4] - nu * PP[4] - mu * PP[4]  # Rn
        Y[5] = gamma * PP[3] - alpha * PP[5] + omega * \
            PP[4] - nu * PP[5] - mu * PP[5]  # Rb

        return Y

    def run(self, IC, t_start, t_end, t_step=1, t_eval=True, events=[]):
        """
        Run the model and store data and time

        TO ADD: next gen matrix, equilibrium

        Parameters
        ----------
        IC : TYPE
            Initial condition vector
        t_start : TYPE
            starting time
        t_end : TYPE
            end time
        t_step : TYPE, optional
            time step. The default is 1.
        t_eval : TYPE, optional
            logical: do we evaluate for all time. The default is True.
        events:
            Can pass in a list of events to go to solve_ivp, i.e. stopping conditions

        Returns
        -------
        self with new data added

        """
        if t_eval:
            t_range = np.arange(
                start=t_start, stop=t_end + t_step, step=t_step)
            self.t_range = t_range

            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            t_eval=t_range,
                            events=events)
            self.results = res.y.T
        else:
            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            events=events)
            self.results = res.y.T

    def get_B(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, 1:6:2], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_S(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, 0:2], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_I(self):
        if hasattr(self, 'results'):
            return np.sum(self.results[:, 2:4], 1)
        else:
            print("Model has not been run")
            return np.nan

    def endemic_behaviour(self, get_res=False):
        """
        Calculate the equilibrium for N, the non-beahviour state.  We then have
        Bstar = 1-Nstar

        Parameters
        ----------
        get_res : TYPE, optional
            Do we want to return the result outside the model. The default is False.

        Returns
        -------
        TYPE
            Equilibirum for the no-behaviour state

        """
        if hasattr(self, 'results'):
            Istar = self.results[-1, 2:4].sum()

            C = self.B_social - self.N_social
            D = self.N_fear * (1 - Istar) + self.N_const + \
                self.B_fear * Istar + self.B_const

            if C == 0:
                self.Nstar = (D - (self.B_fear * Istar + self.B_const)) / D
            else:
                self.Nstar = ((C + D) - np.sqrt((C + D)**2 - 4 * C *
                                                (D - (self.B_fear * Istar + self.B_const)))) / (2 * C)

            if get_res:
                return self.Nstar

        else:
            print("Model has not been run")
            return np.nan

    def NGM(self, get_res=False):
        """
        Calcualte the "behaviour-affected" effective reproduction number using the
        next generation matrix method


        Parameters
        ----------
        CP : array
            Current population proportions.

        Returns
        -------
        Double
            Largest (absolute) eigenvalue for NGM.

        """
        if hasattr(self, 'results'):
            P = self.results[0, 0:6].sum()
            Ib = self.results[:, 3] / P
            In = self.results[:, 2] / P

            if self.infectious_period == 0:
                gamma = 0
            else:
                gamma = 1/self.infectious_period
            if self.av_lifespan == 0:
                mu = 0
            else:
                mu = 1/self.av_lifespan

            gamma = gamma + mu

            tot_B_prop = self.get_B()/P
            tot_inf = self.get_I()/P

            omega = self.rate_to_mask(tot_B_prop=tot_B_prop,
                                      tot_inf=tot_inf)
            alpha = self.rate_to_no_mask(tot_no_B_prop=1 - tot_B_prop,
                                         tot_uninf=1 - tot_inf)

            x = omega + self.B_fear * In - (self.N_social - self.N_fear) * Ib
            y = alpha - (self.B_social + self.B_fear) * In - self.N_fear * Ib

            a = gamma + y + (1 - self.inf_B_efficacy) * x
            b = y + (1 - self.inf_B_efficacy) * (gamma + x)

            Gamma = self.transmission/(gamma * (gamma + x + y))

            self.BA_Reff = Gamma * \
                (self.results[:, 0] * a + (1 - self.susc_B_efficacy)
                 * b * self.results[:, 1]) / P

            if get_res:
                return self.BA_Reff
        else:
            print("Model has not been run")
            return np.nan


def early_behaviour_dynamics(model: bad, method="exp"):
    """
    Early stage dynamics of behaviour

    TO BE IMPLEMENTED
        - methods other than exp
        - methods for k2 not 0
        - methods for if model is not run

    Parameters
    ----------
    model : bad
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is "exp".

    Returns
    -------
    None.

    """
    if hasattr(model, 'results'):
        tt = model.t_range
    else:
        print("Model has not been run")
        return np.nan

    k3 = model.B_social - model.N_social - model.N_fear - model.N_const
    k2 = model.N_fear
    k1 = model.B_fear
    k0 = model.B_const

    I0 = model.results[0, 2:4].sum()

    if k2 != 0.0:
        print("Method currently not implemented")
        return np.nan

    bt = (k0/k3) * (np.exp(k3*tt) - 1)
    it = ((I0 * k1) / (model.transmission - 1/model.infectious_period - k3)) * \
        (np.exp((model.transmission - 1/model.infectious_period) * tt) - np.exp(k3 * tt))

    return bt + it

# %%


if __name__ == "__main__":
    P = 1
    Ib0, Rb0, Rn0 = np.zeros(3)
    Sb0 = 1e-6  # 1 in a million seeded with behaviour
    In0 = 1e-6  # 1 in a million seeded with disease

    Sn0 = P - Sb0 - Ib0 - Rb0 - In0 - Rn0

    PP = np.array([Sn0, Sb0, In0, Ib0, Rn0, Rb0])

    w1 = 8
    R0 = 2
    gamma = 0.4

    cust_params = dict()
    cust_params["transmission"] = R0*gamma
    cust_params["infectious_period"] = 1/gamma
    cust_params["immune_period"] = 240
    cust_params["av_lifespan"] = 0  # Turning off demography
    cust_params["susc_B_efficacy"] = 0.8
    cust_params["inf_B_efficacy"] = 0.4
    cust_params["N_social"] = 0.5
    cust_params["N_fear"] = 0
    cust_params["B_social"] = 0.05 * w1
    cust_params["B_fear"] = w1
    cust_params["B_const"] = 0.01
    cust_params["N_const"] = 0.01

    M1 = bad(**cust_params)

    M1.run(IC=PP, t_start=0, t_end=365, t_step=0.1)

    M1.endemic_behaviour()

    M1.NGM()

    plt.figure()
    plt.title("Dynamics of each strata")
    plt.plot(M1.results[:, 0], "y", label="Sn")
    plt.plot(M1.results[:, 1], "y:", label="Sb")
    plt.plot(M1.results[:, 2], "g", label="In")
    plt.plot(M1.results[:, 3], "g:", label="Ib")
    plt.plot(M1.results[:, 4], "r", label="Rn")
    plt.plot(M1.results[:, 5], "r:", label="Rb")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Disease dynamics")
    plt.plot(M1.get_S(), "y", label="S")
    plt.plot(M1.get_I(), "g", label="I")
    plt.plot(P-M1.get_S()-M1.get_I(), "r", label="R")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Behaviour dynamics w/ endemic behaviour equilibrium")
    plt.plot(M1.t_range, M1.get_B(), "b", label="B")
    plt.plot(M1.t_range, 1-M1.get_B(), "orange", label="N")
    plt.plot([M1.t_range[0], M1.t_range[-1]], [M1.Nstar, M1.Nstar], ":k")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Behaviour affected effective reproduction number")
    plt.plot(M1.t_range, M1.BA_Reff)
    plt.plot([M1.t_range[0], M1.t_range[-1]], [1, 1], ":k")
    plt.xlabel("Time")
    plt.ylabel("BA_Reff")
    plt.show()

    plt.figure()
    plt.title("Disease and behaviour phase plane")
    plt.plot(M1.get_B(), M1.get_I())
    plt.xlabel("Proportion of behaviour")
    plt.ylabel("Proportion of infeceted")
    plt.show()
# %%
    B_est = early_behaviour_dynamics(model=M1)
    tt_stop = [i for i in range(len(B_est)) if B_est[i] <= 1][-1]

    plt.figure()
    plt.title(f"R0 = {R0}")
    plt.plot(M1.t_range[0:tt_stop], M1.get_B()[0:tt_stop],
             "b", label="True behaviour dynamics")
    plt.plot(M1.t_range[0:tt_stop], B_est[0:tt_stop],
             "r:", label="exponential estimate")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
