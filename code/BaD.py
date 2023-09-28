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
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import json
import os
import time

# working_path = "/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel"
# os.chdir(working_path)

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
        self.N_fear = 0

    def set_defaults(self, filename="/Users/rya200/Library/CloudStorage/OneDrive-CSIRO/Documents/03_projects/reid-mask_sir_toymodel/data/BaD_parameter_ranges.json"):
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
        self.N_fear = 0

    def rate_to_infect(self, Ib, In):
        return self.transmission * (In + (1 - self.inf_B_efficacy) * Ib)

    def rate_to_mask(self, tot_B_prop, tot_inf):
        return self.B_social * (tot_B_prop) + self.B_fear * (tot_inf) + self.B_const

    def rate_to_no_mask(self, tot_no_B_prop, tot_uninf):
        return self.N_social * (tot_no_B_prop) + self.N_const

    def odes(self, t, PP, incidence=False):
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

        # NB: Put in parameter checks
        # if incidence:
        #     Y = np.zeros((len(PP) + 1))
        # else:
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
        if incidence:
            Y[6] = lam * PP[0] + lam * (1-self.susc_B_efficacy) * PP[1]

        return Y

    def run(self, IC, t_start, t_end, t_step=1, t_eval=True, events=[], incidence=False):
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
        if incidence:
            args = [incidence]
            IC = np.concatenate((IC, [0]))
        else:
            args = []
        if t_eval:
            t_range = np.arange(
                start=t_start, stop=t_end + t_step, step=t_step)
            self.t_range = t_range

            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            t_eval=t_range,
                            events=events,
                            args=args
                            # rtol=1e-7, atol=1e-14
                            )
            self.results = res.y.T
        else:
            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            events=events,
                            args=args)
            self.results = res.y.T
        if incidence:
            self.incidence = self.results[:, -1]
            self.results = self.results[:, 0:-1]

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

    def endemic_behaviour(self, get_res=False, save=True, I_eval=np.nan):
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
        if not np.isnan(I_eval):
            Istar = I_eval
        elif hasattr(self, 'results'):
            Istar = self.results[-1, 2:4].sum()
        else:
            print("Model has not been run")
            return np.nan

        C = self.B_social - self.N_social
        D = self.N_const + \
            self.B_fear * Istar + self.B_const

        if C == 0:
            if D == 0:
                if hasattr(self, 'results'):
                    Nstar = self.results[0, 0]
                else:
                    Nstar = 1
            else:
                Nstar = (D - (self.B_fear * Istar + self.B_const)) / D
        else:
            Nstar = ((C + D) - np.sqrt((C + D)**2 - 4 * C *
                                       (D - (self.B_fear * Istar + self.B_const)))) / (2 * C)

        if get_res:
            return Nstar
        if save:
            self.Nstar = Nstar

    def NGM(self, get_res=False, orig=False):
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

            if orig:
                Ib = self.results[:, 3] / P
                In = self.results[:, 2] / P
                tot_inf = self.get_I()/P
            else:
                Ib = 0  # self.results[:, 3] / P
                In = 0  # self.results[:, 2] / P
                tot_inf = 0  # self.get_I()/P

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

            omega = self.rate_to_mask(tot_B_prop=tot_B_prop,
                                      tot_inf=tot_inf)
            alpha = self.rate_to_no_mask(tot_no_B_prop=1 - tot_B_prop,
                                         tot_uninf=1 - tot_inf)

            x = omega + self.B_fear * In - (self.N_social) * Ib
            y = alpha - (self.B_social + self.B_fear) * In

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

    def Rzero(self):

        NN = self.endemic_behaviour(get_res=True, save=False, I_eval=0)

        BB = 1-NN

        if self.infectious_period == 0:
            gamma = 0
        else:
            gamma = 1/self.infectious_period
        if self.av_lifespan == 0:
            mu = 0
        else:
            mu = 1/self.av_lifespan

        gamma = gamma + mu

        omega = self.rate_to_mask(tot_B_prop=BB,
                                  tot_inf=0)
        alpha = self.rate_to_no_mask(tot_no_B_prop=NN,
                                     tot_uninf=1)

        x = omega
        y = alpha

        a = gamma + y + (1 - self.inf_B_efficacy) * x
        b = y + (1 - self.inf_B_efficacy) * (gamma + x)

        Gamma = self.transmission/(gamma * (gamma + x + y))

        BA_R0 = Gamma * \
            (NN * a + (1 - self.susc_B_efficacy)
             * b * BB)
        return BA_R0

# %% functions external to class


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

    k3 = model.B_social - model.N_social - model.N_const
    k2 = 0
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

# todo: Can I implement the taylor coefficients automatically?


def calculate_F_m(t, M, model):
    k3 = model.B_social - model.N_social - model.N_const

    if M == 0:
        return (np.exp(k3 * t) - 1)/k3
    else:
        return (M/k3) * calculate_F_m(t=t, M=M-1, model=model) - (t**M)/k3

# The next set of functions are aimed towards calculating the steady states of the system


def get_B_a_w(I, params):
    """
    Cacluate the steady state of behaviour, and alpha and omega values at steady state when I is known.

    Parameters
    ----------
    I : float
        Desired prevalence of disease at equilibrium
    params : dict
        dictionary of model parameters.  Must include:
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)
                "transmission" (beta)
                "inf_B_efficacy" (p)

    Returns
    -------
    B : float
        Proportion of the population performing the behaviour at equilibrium
    a : float
        alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
    w : float
        omega transition value at equilibrium (w1 * B + w2 * I + w3)
    """

    nu = 1/params["immune_period"]
    g = 1/params["infectious_period"]

    assert not (I > nu/(g + nu)), "invalid choice of I"

    if I < 1e-8:
        I = 0

    D = params["N_const"] + params["B_fear"] * I + params["B_const"]
    C = params["B_social"] - params["N_social"]

    if C == 0:
        if D == 0:
            N = 1
        else:
            N = (D - (params["B_fear"] * I + params["B_const"])) / D
    else:
        N = ((C + D) - np.sqrt((C + D)**2 - 4 * C *
             (D - (params["B_fear"] * I + params["B_const"])))) / (2 * C)
    B = 1 - N
    a = params["N_social"] * (1-B) + params["N_const"]
    w = params["B_social"] * B + params["B_fear"] * I + params["B_const"]

    return B, a, w


def get_R_S(I, params):
    """
    Cacluate the steady state of R ans S when I is known.

    Parameters
    ----------
    I : float
        Desired prevalence of disease at equilibrium
    params : dict
        dictionary of model parameters.  Must include:
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)
                "transmission" (beta)
                "inf_B_efficacy" (p)

    Returns
    -------
    R : float
        Proportion of the population recovered at equilibrium
    S : float
        Proportion of the population susceptible at equilibrium
    """
    g = 1/params["infectious_period"]
    nu = 1/params["immune_period"]
    R = g/nu * I
    S = 1 - I - R
    return R, S


# def get_lambda(S, I, B, a, w, params):
#     """
#     Calculate the force of infection at equilibrium

#     Parameters
#     ----------
#     S : float
#         Proportion of the population susceptible at equilibrium
#     I : float
#         Desired prevalence of disease at equilibrium
#     B : float
#         Proportion of the population performing the behaviour at equilibrium
#     a : float
#         alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
#     w : float
#         omega transition value at equilibrium (w1 * B + w2 * I + w3)
#     params : dict
#         dictionary of model parameters.  Must include:
#                 "infectious_period" (1/gamma)
#                 "immune_period" (1/nu)
#                 "susc_B_efficacy" (c)
#                 "N_social" (a1)
#                 "N_fear" (a2)
#                 "N_const" (a3)
#                 "B_social" (w1)
#                 "B_fear" (w2)
#                 "B_const" (w3)
#         Optional parameters are:
#                 "transmission" (beta)
#                 "inf_B_efficacy" (p)

#     Returns
#     -------
#     lam : float
#         Force of infection at equilibrium
#     """

#     N = 1-B

#     g = 1/params["infectious_period"]
#     v = 1/params["immune_period"]
#     c = params["susc_B_efficacy"]

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
#     """
#     Calculate the proportion of the population in Sb at equilibrium

#     Parameters
#     ----------
#     S : float
#         Proportion of the population susceptible at equilibrium
#     I : float
#         Desired prevalence of disease at equilibrium
#     B : float
#         Proportion of the population performing the behaviour at equilibrium
#     a : float
#         alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
#     w : float
#         omega transition value at equilibrium (w1 * B + w2 * I + w3)
#     params : dict
#         dictionary of model parameters.  Must include:
#                 "infectious_period" (1/gamma)
#                 "immune_period" (1/nu)
#                 "susc_B_efficacy" (c)
#                 "N_social" (a1)
#                 "N_fear" (a2)
#                 "N_const" (a3)
#                 "B_social" (w1)
#                 "B_fear" (w2)
#                 "B_const" (w3)
#         Optional parameters are:
#                 "transmission" (beta)
#                 "inf_B_efficacy" (p)

#     Returns
#     -------
#     sb : float
#         Proportion of the population in Sb at equilibrium
#     """

#     g = 1/params["infectious_period"]
#     v = 1/params["immune_period"]
#     c = params["susc_B_efficacy"]

#     if c == 0:
#         detA = (lam + a + w + v) * (a + w + g) + lam*v
#         sb = (a + w + g) * (w * S + v * B) - w * v * I
#         sb /= detA
#     else:
#         sb = (lam*S - g * I)/(c * lam)
#     return sb


# def get_In(lam, S, I, B, a, w, params):
#     """
#     Calculate the proportion of the population in In at equilibrium

#     Parameters
#     ----------
#     S : float
#         Proportion of the population susceptible at equilibrium
#     I : float
#         Desired prevalence of disease at equilibrium
#     B : float
#         Proportion of the population performing the behaviour at equilibrium
#     a : float
#         alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
#     w : float
#         omega transition value at equilibrium (w1 * B + w2 * I + w3)
#     params : dict
#         dictionary of model parameters.  Must include:
#                 "infectious_period" (1/gamma)
#                 "immune_period" (1/nu)
#                 "susc_B_efficacy" (c)
#                 "N_social" (a1)
#                 "N_fear" (a2)
#                 "N_const" (a3)
#                 "B_social" (w1)
#                 "B_fear" (w2)
#                 "B_const" (w3)
#         Optional parameters are:
#                 "transmission" (beta)
#                 "inf_B_efficacy" (p)

#     Returns
#     -------
#     In : float
#         Proportion of the population in In at equilibrium
#     """

#     g = 1/params["infectious_period"]
#     v = 1/params["immune_period"]
#     c = params["susc_B_efficacy"]

#     if c == 0:
#         detA = (lam + a + w + v) * (a + w + g) + lam*v
#         Ib = lam * (w*S + v*B) + (lam + a + w + v) * w * I
#         Ib /= detA
#         In = I - Ib
#     else:
#         In = (g + c * a) * \
#             I - (1 - c) * lam * S
#         In = In/(c * (a + w + g))

#     return In


# def get_steady_states(I, params):
#     """
#     Calculate the steady state vector for a given disease prevalence and set of parameters.
#     Also returns the force of infection at equilibrium.
#     Parameters
#     ----------
#     I : float
#         Desired prevalence of disease at equilibrium
#     params : dict
#         dictionary of model parameters.  Must include:
#                 "infectious_period" (1/gamma)
#                 "immune_period" (1/nu)
#                 "susc_B_efficacy" (c)
#                 "N_social" (a1)
#                 "N_fear" (a2)
#                 "N_const" (a3)
#                 "B_social" (w1)
#                 "B_fear" (w2)
#                 "B_const" (w3)
#         Optional parameters are:
#                 "transmission" (beta)
#                 "inf_B_efficacy" (p)

#     Returns
#     -------
#     ss : numpy.array
#         Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
#     lam : float
#         For of infection at equilibrium.

#     """

#     B, a, w = get_B_a_w(I, params)

#     if I <= 0:
#         lam = 0
#         return np.array([1-B, B, 0.0, 0.0, 0.0, 0.0]), lam

#     R, S = get_R_S(I, params)

#     lam = get_lambda(S, I, B, a, w, params)

#     Sb = get_Sb(lam, S, I, B, a, w, params)
#     Sn = S - Sb

#     In = get_In(lam, S, I, B, a, w, params)
#     Ib = I - In

#     Rn = (1-B) - Sn - In

#     Rb = B - Sb - Ib

#     ss = np.array([Sn, Sb, In, Ib, Rn, Rb])

#     return ss, lam


# def solve_I(i, params):
#     """
#     Function to numerically find the disease prevalence at equilibrium for a given set
#     of model parameters.  Designed to be used in conjunction with fsolve.

#     Parameters
#     ----------
#     i : float
#         Estimated disease prevalence at equilibrium.
#     params : dict
#         dictionary of model parameters.  Must include:
#                 "transmission" (beta)
#                 "infectious_period" (1/gamma)
#                 "immune_period" (1/nu)
#                 "susc_B_efficacy" (c)
#                 "inf_B_efficacy" (p)
#                 "N_social" (a1)
#                 "N_fear" (a2)
#                 "N_const" (a3)
#                 "B_social" (w1)
#                 "B_fear" (w2)
#                 "B_const" (w3)
#     Returns
#     -------
#     res : float
#         The difference between the model predicted force of infection (assuming I* = i) and the theoretical
#         lambda = beta * (I - p Ib).
#     """

#     assert "transmission" in params.keys(), "define transmission"

#     assert "inf_B_efficacy" in params.keys(), "define inf_B_efficacy"

#     B, a, w = get_B_a_w(i, params)

#     R, S = get_R_S(i, params)

#     lam = get_lambda(S, i, B, a, w, params)

#     In = get_In(lam, S, i, B, a, w, params)
#     Ib = i - In

#     res = params["transmission"]*(i - params["inf_B_efficacy"] * Ib) - lam

#     return res


# def find_ss(params):
#     """
#     Calculate the steady states of the system for a given set of model parameters.  We first numerically
#     solve for the disease prevalence I, then use I to find all other steady states.

#     Parameters
#     ----------
#     params : dict
#         dictionary of model parameters.  Must include:
#                 "transmission" (beta)
#                 "infectious_period" (1/gamma)
#                 "immune_period" (1/nu)
#                 "susc_B_efficacy" (c)
#                 "inf_B_efficacy" (p)
#                 "N_social" (a1)
#                 "N_fear" (a2)
#                 "N_const" (a3)
#                 "B_social" (w1)
#                 "B_fear" (w2)
#                 "B_const" (w3)

#     Returns
#     -------
#     ss : numpy.array
#         Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
#     lam : float
#         For of infection at equilibrium.
#     Istar : float
#         Estimated disease prevalence at equilibrium
#     """

#     nu = 1/params["immune_period"]
#     g = 1/params["infectious_period"]

#     init_i = nu/(g + nu) - 1e-3

#     Istar = fsolve(solve_I, x0=[init_i], args=(params))

#     if Istar[0] < 1e-8:
#         Istar[0] = 0

#     ss, lam = get_steady_states(Istar[0], params)

#     return ss, lam, Istar[0]


def get_Ib(I, S, a, w, params):
    """
    Calculate the proportion of people infected and performing the behaviour

    Parameters
    ----------
    I : float
        Desired prevalence of disease at equilibrium
    S : float
        Prevalence of susceptible at equilibrium
    a : float
        alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
    w : float
        omega transition value at equilibrium (w1 * B + w2 * I + w3)
    params : dict
        dictionary of model parameters.  Must include:
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)
                "transmission" (beta)
                "inf_B_efficacy" (p)

    Returns
    -------
    ib : float
        Prevalence of infected performing the behaviour.

    """
    p = params["inf_B_efficacy"]
    c = params["susc_B_efficacy"]
    beta = params["transmission"]
    g = 1/params["infectious_period"]

    if (np.isclose(p, 0)) and (np.isclose(c, 0)):
        B, _, _ = get_B_a_w(I, params)
        v = 1/params["immune_period"]
        R = 1-S-I
        numer = (beta*(B*(w+a+v)-w*R) + w*(w+a+v)) * I
        denom = (w+a+g+beta*I)*(a+w+v) + g
    else:
        numer = (1-c) * (beta * S - g) * I + c*w*I
        denom = (1-c) * p * beta * S + c * (a + w + g)

    ib = numer/denom

    return ib


def get_Rb(R, Ib, a, w, params):
    """
    Calculate proportion of recovereds doing behaviour.

    Parameters
    ----------
    R : float
        Proportion of individuals who are recovered.
    Ib : float
        Proportion of infected individuals doing the behaviour.
    a : float
        alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
    w : float
        omega transition value at equilibrium (w1 * B + w2 * I + w3)
    params : dict
        dictionary of model parameters.  Must include:
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)
                "transmission" (beta)
                "inf_B_efficacy" (p)

    Returns
    -------
    Rb : float
        Proportion of recvoered individuals doing the behaviour.

    """
    g = 1/params["infectious_period"]
    v = 1/params["immune_period"]

    Rb = (w*R + g*Ib)/(a+w+v)

    return Rb


def get_steady_states(I, params):
    """
    Calculate the steady state vector for a given disease prevalence and set of parameters.  
    Parameters
    ----------
    I : float
        Desired prevalence of disease at equilibrium
    params : dict
        dictionary of model parameters.  Must include:
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)
                "transmission" (beta)
                "inf_B_efficacy" (p)

    Returns
    -------
    ss : numpy.array
        Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
    """

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
    """
    Function to numerically find the disease prevalence at equilibrium for a given set
    of model parameters.  Designed to be used in conjunction with fsolve.

    Parameters
    ----------
    i : float
        Estimated disease prevalence at equilibrium.
    params : dict
        dictionary of model parameters.  Must include:
                "transmission" (beta)
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "inf_B_efficacy" (p)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)
    Returns
    -------
    res : float
        The difference between the (lambda + w)*S_n and a*S_b + nu*R_n.
    """

    B, a, w = get_B_a_w(i, params)

    ss_n = get_steady_states(i, params)

    lam = params["transmission"] * \
        (ss_n[2] + (1-params["inf_B_efficacy"]) * ss_n[3])

    res = (lam + w) * ss_n[0] - (a * ss_n[1] +
                                 1/params["immune_period"] * ss_n[4])

    return res


def find_ss(params):
    """
    Calculate the steady states of the system for a given set of model parameters.  We first numerically
    solve for the disease prevalence I, then use I to find all other steady states.

    Parameters
    ----------
    params : dict
        dictionary of model parameters.  Must include:
                "transmission" (beta)
                "infectious_period" (1/gamma)
                "immune_period" (1/nu)
                "susc_B_efficacy" (c)
                "inf_B_efficacy" (p)
                "N_social" (a1)
                "N_fear" (a2)
                "N_const" (a3)
                "B_social" (w1)
                "B_fear" (w2)
                "B_const" (w3)

    Returns
    -------
    ss : numpy.array
        Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
    Istar : float
        Estimated disease prevalence at equilibrium
    """

    nu = 1/params["immune_period"]
    g = 1/params["infectious_period"]

    # deal with case when p = c = 0
    if (np.isclose(params["susc_B_efficacy"], 0)) and (np.isclose(params["inf_B_efficacy"], 0)):
        Istar = [nu * (params["transmission"] - g) /
                 (params["transmission"] * (g + nu))]
    else:
        init_i = nu/(g + nu) - 1e-3
        Istar = fsolve(solve_I, x0=[init_i], args=(params))

    if Istar[0] < 1e-8:
        Istar[0] = 0

    ss = get_steady_states(Istar[0], params)

    return ss, Istar[0]


# %%


if __name__ == "__main__":
    P = 1
    Ib0, Rb0, Rn0 = np.zeros(3)
    Sb0 = 1e-6  # 1 in a million seeded with behaviour
    In0 = 1e-6  # 1 in a million seeded with disease
    # Ib0, Rb0, Rn0 = np.zeros(3)
    # Sb0 = 1-0.6951793156273507  # 1 in a million seeded with behaviour
    # In0 = Ib0 = 1e-6  # 1 in a million seeded with disease

    Sn0 = P - Sb0 - Ib0 - Rb0 - In0 - Rn0

    PP = np.array([Sn0, Sb0, In0, Ib0, Rn0, Rb0])

    w1 = 8
    R0 = 5
    gamma = 1/5

    cust_params = dict()
    cust_params["transmission"] = R0*gamma
    cust_params["infectious_period"] = 1/gamma
    cust_params["immune_period"] = 240
    cust_params["av_lifespan"] = 0  # Turning off demography
    cust_params["susc_B_efficacy"] = 0.4
    cust_params["inf_B_efficacy"] = 0.4
    cust_params["N_social"] = 0.5
    cust_params["B_social"] = 0.05 * w1
    cust_params["B_fear"] = 0.01  # w1
    cust_params["B_const"] = 0.0
    cust_params["N_const"] = 0.01
    # cust_params["transmission"] = R0*gamma
    # cust_params["infectious_period"] = 1/gamma
    # cust_params["immune_period"] = 240
    # cust_params["av_lifespan"] = 0  # Turning off demography
    # cust_params["susc_B_efficacy"] = 0.8
    # cust_params["inf_B_efficacy"] = 0.4
    # cust_params["N_social"] = 0.
    # cust_params["B_social"] = 0.0
    # cust_params["B_fear"] = 0
    # cust_params["B_const"] = 0.0
    # cust_params["N_const"] = 0.0

    M1 = bad(**cust_params)

    M1.run(IC=PP, t_start=0, t_end=900, t_step=1)

    M1.endemic_behaviour()

    M1.NGM()

    tt = M1.t_range[0:(len(M1.results))]

    plt.figure()
    plt.title("Dynamics of each strata")
    plt.plot(tt, M1.results[:, 0], "y", label="Sn")
    plt.plot(tt, M1.results[:, 1], "y:", label="Sb")
    plt.plot(tt, M1.results[:, 2], "g", label="In")
    plt.plot(tt, M1.results[:, 3], "g:", label="Ib")
    plt.plot(tt, M1.results[:, 4], "r", label="Rn")
    plt.plot(tt, M1.results[:, 5], "r:", label="Rb")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Disease dynamics")
    # plt.plot(tt, M1.get_S(), "y", label="S")
    plt.plot(tt, M1.get_I(), "g", label="I")
    # plt.plot(tt, P-M1.get_S()-M1.get_I(), "r", label="R")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Behaviour dynamics w/ endemic behaviour equilibrium")
    plt.plot(tt, M1.get_B(), "b", label="B")
    plt.plot(tt, 1-M1.get_B(), "orange", label="N")
    plt.plot([tt[0], tt[-1]], [M1.Nstar, M1.Nstar], ":k")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Behaviour affected effective reproduction number")
    plt.plot(tt, M1.BA_Reff)
    plt.plot([tt[0], tt[-1]], [1, 1], ":k")
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
    # tt_stop = [i for i in range(len(B_est)) if B_est[i] <= 1][-1]
    tt_stop = next((i for i in range(len(B_est)) if B_est[i] > 1))

    plt.figure()
    plt.title(f"R0 = {R0}")
    plt.plot(tt[0:tt_stop], M1.get_B()[0:tt_stop],
             "b", label="True behaviour dynamics")
    plt.plot(tt[0:tt_stop], B_est[0:tt_stop],
             "r:", label="exponential estimate")
    plt.xlabel("Time")
    plt.legend()
    plt.show()

# %%
    ss, _ = find_ss(cust_params)

    print(
        f"Numeric Sn* = {M1.results[-1, 0]}, estimated Sn* = {ss[0]}, absolute difference = {np.abs(M1.results[-1, 0] - ss[0])}")
    print(
        f"Numeric Sb* = {M1.results[-1, 1]}, estimated Sb* = {ss[1]}, absolute difference = {np.abs(M1.results[-1, 1] - ss[1])}")
    print(
        f"Numeric In* = {M1.results[-1, 2]}, estimated In* = {ss[2]}, absolute difference = {np.abs(M1.results[-1, 2] - ss[2])}")
    print(
        f"Numeric Ib* = {M1.results[-1, 3]}, estimated Ib* = {ss[3]}, absolute difference = {np.abs(M1.results[-1, 3] - ss[3])}")
    print(
        f"Numeric Rn* = {M1.results[-1, 4]}, estimated Rn* = {ss[4]}, absolute difference = {np.abs(M1.results[-1, 4] - ss[4])}")
    print(
        f"Numeric Rb* = {M1.results[-1, 5]}, estimated Rb* = {ss[5]}, absolute difference = {np.abs(M1.results[-1, 5] - ss[5])}")

    print("\n")

    print(
        f"Numeric S* = {M1.results[-1, [0,1]].sum()}, estimated S* = {ss[[0, 1]].sum()}, absolute difference = {np.abs(M1.results[-1, [0, 1]].sum() - ss[[0,1]].sum())}")
    print(
        f"Numeric I* = {M1.results[-1, [2,3]].sum()}, estimated I* = {ss[[2, 3]].sum()}, absolute difference = {np.abs(M1.results[-1, [2, 3]].sum() - ss[[2,3]].sum())}")
    print(
        f"Numeric R* = {M1.results[-1, [4,5]].sum()}, estimated R* = {ss[[4, 5]].sum()}, absolute difference = {np.abs(M1.results[-1, [4, 5]].sum() - ss[[4,5]].sum())}")

    print("\n")

    print(f"Numeric B* = {M1.results[-1, [1, 3, 5]].sum()}, estimated B* = {ss[[1, 3, 5]].sum()}, absolute difference = {np.abs(M1.results[-1, [1, 3, 5]].sum() - ss[[1, 3, 5]].sum())}")


# %%
    plt.figure()
    plt.title("Dynamics of susceptibles with predicted steady states")
    plt.plot(tt, M1.results[:, 0], "y", label="Sn")
    plt.plot(tt, M1.results[:, 1], "purple", label="Sb")
    plt.plot([tt[0], tt[-1]], [ss[0], ss[0]], "y:", label="Sn")
    plt.plot([tt[0], tt[-1]], [ss[1], ss[1]],
             "purple", linestyle=":", label="Sb")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Dynamics of infecteds with predicted steady states")
    plt.plot(tt, M1.results[:, 2], "r", label="In")
    plt.plot(tt, M1.results[:, 3], "orange", label="Ib")
    plt.plot([tt[0], tt[-1]], [ss[2], ss[2]], "r:", label="In")
    plt.plot([tt[0], tt[-1]], [ss[3], ss[3]],
             "orange", linestyle=":", label="Ib")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()

    plt.figure()
    plt.title("Dynamics of recovereds with predicted steady states")
    plt.plot(tt, M1.results[:, 4], "b", label="Rn")
    plt.plot(tt, M1.results[:, 5], "lightblue", label="Rb")
    plt.plot([tt[0], tt[-1]], [ss[4], ss[4]], "b:", label="Rn")
    plt.plot([tt[0], tt[-1]], [ss[5], ss[5]],
             "lightblue", linestyle=":", label="Rb")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Proportion")
    plt.show()


# %% Create bifurcation plot alla Mick Roberts

    bifurc_params = dict()
    bifurc_params["transmission"] = 1
    bifurc_params["infectious_period"] = 1/1
    bifurc_params["immune_period"] = 1/0.5
    bifurc_params["av_lifespan"] = 0  # Turning off demography
    bifurc_params["susc_B_efficacy"] = 0.5
    bifurc_params["inf_B_efficacy"] = 0.3
    bifurc_params["N_social"] = 0.2
    bifurc_params["B_social"] = 1.3
    bifurc_params["B_fear"] = 0.5
    bifurc_params["B_const"] = 0.7
    bifurc_params["N_const"] = 0.9

    # Different set of params
    # bifurc_params = dict()
    # bifurc_params["transmission"] = 1
    # bifurc_params["infectious_period"] = 1/0.4
    # bifurc_params["immune_period"] = 1/(8*30)
    # bifurc_params["av_lifespan"] = 0  # Turning off demography
    # bifurc_params["susc_B_efficacy"] = 0.4
    # bifurc_params["inf_B_efficacy"] = 0.8
    # bifurc_params["N_social"] = 0.5
    # bifurc_params["B_social"] = 0.05 * 8
    # bifurc_params["B_fear"] = 8
    # bifurc_params["B_const"] = 0.01
    # bifurc_params["N_const"] = 0.01

    M2 = bad(**bifurc_params)
    NN = M2.endemic_behaviour(get_res=True, save=False, I_eval=0)

    a = bifurc_params["N_social"] * NN + bifurc_params["N_const"]
    w = bifurc_params["B_social"] * (1-NN) + bifurc_params["B_const"]

    multi_val = (1/bifurc_params["infectious_period"])*(1/bifurc_params["infectious_period"] + a + w) / (NN * (a + 1/bifurc_params["infectious_period"] + (
        1 - bifurc_params["inf_B_efficacy"]) * w) + (1-bifurc_params["susc_B_efficacy"]) * (1-NN) * (a + (1 - bifurc_params["inf_B_efficacy"]) * (1/bifurc_params["infectious_period"] + w)))
    # multi_val = 1/bifurc_params["infectious_period"]

    beta_vals = np.arange(1., 3.1, step=0.1)

    res = list()

    for idxx in range(len(beta_vals)):
        b = beta_vals[idxx] * multi_val
        bifurc_params["transmission"] = b
        ss, _ = find_ss(bifurc_params)

        # ss[ss.round(4) > 0] = 1
        res.append(ss.round(4))

    res = np.array(res)

    plt.figure()
    plt.ylim([0, 1])
    plt.plot(beta_vals, res[:, 0], "green", label="Sn")
    plt.plot(beta_vals, res[:, 1], "black", label="Sb")
    plt.plot(beta_vals, res[:, 2], "red", label="In")
    plt.plot(beta_vals, res[:, 3], "magenta", label="Ib")
    plt.plot(beta_vals, res[:, 4], "blue", label="Rn")
    plt.plot(beta_vals, res[:, 5], "cyan", label="Rb")
    plt.legend()
    plt.show()

# %% Heat maps

    epi_r0 = True  # Plot beta/gamma on x axis or not

    heat_map_params = dict()
    heat_map_params["transmission"] = 1
    heat_map_params["infectious_period"] = 1/1
    heat_map_params["immune_period"] = 1/0.5
    heat_map_params["av_lifespan"] = 0  # Turning off demography
    heat_map_params["susc_B_efficacy"] = 0.5
    heat_map_params["inf_B_efficacy"] = 0.3
    heat_map_params["N_social"] = 0.2
    heat_map_params["B_social"] = 1.3
    heat_map_params["B_fear"] = 0.5
    heat_map_params["B_const"] = 0.7
    heat_map_params["N_const"] = 0.9

    # Different set of params
    # heat_map_params = dict()
    # heat_map_params["transmission"] = 1
    # heat_map_params["infectious_period"] = 1/0.4
    # heat_map_params["immune_period"] = 1/(8*30)
    # heat_map_params["av_lifespan"] = 0  # Turning off demography
    # heat_map_params["susc_B_efficacy"] = 0.4
    # heat_map_params["inf_B_efficacy"] = 0.8
    # heat_map_params["N_social"] = 0.5
    # heat_map_params["B_social"] = 0.05 * 8
    # heat_map_params["B_fear"] = 8
    # heat_map_params["B_const"] = 0.01
    # heat_map_params["N_const"] = 0.01

    epi_r0_range = np.arange(0.1, 8.1, step=0.1)
    behav_r0_range = np.arange(0.1, 8.1,  step=0.1)

    xx, yy = np.meshgrid(epi_r0_range, behav_r0_range)

    r0_combos = np.array(np.meshgrid(epi_r0_range, behav_r0_range)).reshape(
        (2, len(epi_r0_range) * len(behav_r0_range))).T

    ss_list = list()
    R0_list = list()

    M3 = bad(**heat_map_params)

    for idx in range(len(r0_combos)):
        b = r0_combos[idx, 0]
        w = r0_combos[idx, 1]
        ww = w * (heat_map_params["N_social"] + heat_map_params["N_const"])

        if epi_r0:
            bb = b * (1/heat_map_params["infectious_period"])
        else:
            M3.update_params(**{"B_social": ww})
            inter_r0 = M3.Rzero()
            multi_val2 = inter_r0/M3.transmission
            bb = b * (1/multi_val2)

        heat_map_params["B_social"] = ww
        heat_map_params["transmission"] = bb

        ss, _ = find_ss(heat_map_params)
        ss_list.append(ss)

        M3.update_params(**{"transmission": bb, "B_social": ww})
        R0_list.append(M3.Rzero())

    def calc_B(X):
        xx = X[1]
        B = xx[[1, 3, 5]].sum()
        return B

    def calc_I(X):
        xx = X[1]
        B = xx[[2, 3]].sum()
        return B

    BB = list(map(calc_B, enumerate(ss_list)))
    II = list(map(calc_I, enumerate(ss_list)))

    if epi_r0:
        beta_vals = list()
        for idx in range(len(r0_combos)):
            b = r0_combos[idx, 0]
            w = r0_combos[idx, 1]
            bb = b * (1/heat_map_params["infectious_period"])
            ww = w * (heat_map_params["N_social"] + heat_map_params["N_const"])

            M3.update_params(**{"transmission": bb, "B_social": ww})

            tmp = M3.Rzero()/bb

            beta_vals.append(1/tmp)

        new_R0 = np.array(beta_vals) * heat_map_params["infectious_period"]

# %%
    # vec_BB = np.array(II)
    # stp = vec_BB[vec_BB.nonzero()[0]].ptp()/6
    # lvls = np.arange(vec_BB[vec_BB.nonzero()[0]].min(),
    #                  1 + stp, step=stp)
    # lvls = np.concatenate(([0], lvls))

    plt.figure()
    plt.title("Steady state of behaviour")
    plt.contourf(xx, yy, np.array(BB).reshape(xx.shape),
                 # levels = lvls,
                 cmap=plt.cm.Blues)
    plt.colorbar(format=tkr.PercentFormatter(xmax=1, decimals=2))
    if epi_r0:
        plt.xlabel("Epidemic R0")
        plt.plot(new_R0, r0_combos[:, 1], "k:")
    else:
        plt.xlabel("Behaviour affected R0")
    plt.ylabel("Behaviour R0")
    plt.show()

    vec_II = np.array(II)
    stp = vec_II[vec_II.nonzero()[0]].ptp()/6
    lvls = np.arange(vec_II[vec_II.nonzero()[0]].min(),
                     vec_II.max() + stp, step=stp)
    lvls = np.concatenate(([0], lvls))

    plt.figure()
    plt.title("Steady state of Infection")
    plt.contourf(xx, yy, vec_II.reshape(xx.shape),
                 levels=lvls, cmap=plt.cm.Reds)
    # plt.plot(new_R0, r0_combos[:, 1], "k:")
    plt.colorbar(format=tkr.PercentFormatter(xmax=1, decimals=2))
    # plt.xlabel("Epidemic R0")
    if epi_r0:
        plt.xlabel("Epidemic R0")
        plt.plot(new_R0, r0_combos[:, 1], "k:")
    else:
        plt.xlabel("Behaviour affected R0")
    plt.ylabel("Behaviour R0")
    plt.show()

    plt.figure()
    plt.title("Behaviour affected R0")
    plt.contourf(xx, yy, np.array(R0_list).reshape(
        xx.shape),  cmap=plt.cm.Greens)
    # plt.plot(new_R0, r0_combos[:, 1], "k:")
    plt.colorbar()
    # plt.xlabel("Epidemic R0")
    if epi_r0:
        plt.xlabel("Epidemic R0")
        plt.plot(new_R0, r0_combos[:, 1], "k:")
    else:
        plt.xlabel("Behaviour affected R0")
    plt.ylabel("Behaviour R0")
    plt.show()

# %%
