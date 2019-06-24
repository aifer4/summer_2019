#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Settings and Parameters"""

# Plot Settings
plot_params = {'legend.fontsize': 'x-large',
               'figure.figsize': (13, 7),
               'axes.labelsize': 'x-large',
               'axes.titlesize':'x-large',
               'xtick.labelsize':'x-large',
               'ytick.labelsize':'x-large',
               'font.family' :'serif',
               'font.sans-serif' :'Garamond'}


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

# Numerical Settings
τ0 = .001 / (H0 * 1000)

k_low = .0001 *H0*1000
k_high= 2.0  *H0*1000

N = 100
K = 100

