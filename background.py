#!/usr/bin/env python
# coding: utf-8


"""Background Cosmology"""

from params import *
from scipy import integrate
import numpy as np

def τ_itgd(a):
    """conformal time integrand"""
    return  1/(a**2* H0 * np.sqrt(Ωm0*a**-3 + Ωr0*a**-4 + ΩΛ))

def get_τ(a):
    (τ,_)=integrate.quad(τ_itgd,0,a)
    return τ

τ_rec = get_τ(a_rec)
τ_eq = get_τ(a_eq)
τ_now = get_τ(1.0)

ℋ = lambda τ: 2*α*(α*τ/τr + 1)/(α**2 * (τ**2/τr))
a = lambda τ: a_eq*((α*τ/τr)**2 + 2*α*τ/τr)
y = lambda τ: a(τ)/a_eq
yb = lambda τ: 1.68*y(τ)*Ωb0/Ωm0

Ωb = lambda τ: Ωb0 * a(τ)**-3.
Ωc = lambda τ: Ωc0 * a(τ)**-3.
Ωɣ = lambda τ: Ωɣ0 * a(τ)**-4.
Ων = lambda τ: Ων0 * a(τ)**-4.
Ωm = lambda τ: Ωm0 * a(τ)**-3.
Ωr = lambda τ: Ωr0 * a(τ)**-4.
Ωd = lambda τ: Ωc(τ) + Ων(τ)



