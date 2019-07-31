#!/usr/bin/env python
# coding: utf-8
"""
This file sets the numerical values of parameters used by 
the Jupyter notebooks. Some derived parameters are also 
included here for convenience. The parameters are accessed
by using an import statement, which can be done like this:

>>> from params import *
>>> print(H0)

or like this:

>>> import params
>>> print(params.H0)

The basic unit of length [l] and time [t] are specified by
setting the present-day hubble value (H0) and the speed of
light (c) to unity.
[t] ~ (14 Gyr)^-1
[l] ~ 1.4e26 m
"""

import numpy as np

# Parameters
H0 = 1.
Ωb0 = 0.04968
Ωc0 = 0.26754
Ωɣ0 = 5.50234e-5
Ων0 = 3.74248e-5

#normalization params
As = np.exp(3.064)/1.e10
ns = .9667
k_star = 0.05 # Mpc^-1
TCMB0 =2.72548 * 1.e6 # μK
c = 1

Ωd0 = Ων0 + Ωc0
Ωr0 = Ωɣ0 + Ων0
Ωm0 = Ωb0 + Ωc0
ΩΛ  = 1-(Ωm0 + Ωr0)

z_rec = 1090
a_rec = 1/(1+z_rec)
a_eq  = Ωr0/Ωm0

τr = 1/(np.sqrt(Ωm0/a_rec)*H0/2)
τs = 0.6*Ωm0**(.25) * Ωb0**(-.5)*a_rec**(.75)*(H0/100)**(-.5)*τr
τs = 0.6*Ωm0**(.25) * Ωb0**(-.5)*a_rec**(.75)*(.67)**(-.5)*τr
α = np.sqrt(a_rec/a_eq)


# Numerical Settings
τ0 = 1.e-7 # initial time
k_low = 1  # lowest wavenumber
k_high= 1000# highest wavenumber
N = 2**12 # number of points used to solve the ODE
K_SOLVE = 256 # number of wavenumbers for which to solve the  ODE
K_INT = 10000 #number of wavenumbers at which to evaluate bessel functions/integrate Cl


# Plot Settings
plot_params = {'legend.fontsize': 'x-large',
               'figure.figsize': (13, 7),
               'axes.labelsize': 'x-large',
               'axes.titlesize':'x-large',
               'xtick.labelsize':'x-large',
               'ytick.labelsize':'x-large',
               'font.family' :'serif',
               'font.sans-serif' :'Garamond'}


