#!/usr/bin/env python
# coding: utf-8


"""Settings and Parameters"""

import numpy as np

# Parameters
H0 = 67.
Ωb0 = 0.04968
Ωc0 = 0.26754
Ωɣ0 = 5.50234e-5
Ων0 = 3.74248e-5

Ωr0 = Ωɣ0 + Ων0
Ωm0 = Ωb0 + Ωc0
ΩΛ  = 1-(Ωb0 + Ωc0 + Ωɣ0)

z_rec = 1090
a_rec = 1/(1+z_rec)
a_eq  = Ωr0/Ωm0

τr = 1/(np.sqrt(Ωm0/a_rec)*H0/2)
τs = 0.6*Ωm0**(.25) * Ωb0**(-.5)*a_rec**(.75)*(H0/100)**(-.5)*τr
α = np.sqrt(a_rec/a_eq)


# Numerical Settings
τ0 = .001 / (H0 * 1000)

k_low = .1
k_high=100000


N = 100
NMIN = 1000
K_SOLVE = 200
K_INT = 10000


# Plot Settings
plot_params = {'legend.fontsize': 'x-large',
               'figure.figsize': (13, 7),
               'axes.labelsize': 'x-large',
               'axes.titlesize':'x-large',
               'xtick.labelsize':'x-large',
               'ytick.labelsize':'x-large',
               'font.family' :'serif',
               'font.sans-serif' :'Garamond'}


