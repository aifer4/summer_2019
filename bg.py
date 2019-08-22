"""Evaluate background quantities."""

import par
from scipy import integrate
import numpy as np
import numba
import numops

myparams = np.zeros(6 + par.NC + par.NC**2)

A = np.linspace(0, par.a_rec, par.N_SOLVE)

def tau_itgd_ML(a):
    # conformal time integrand valid for tau < tau0_pca (when wd = wd_ML).
    return  1/(a**2* par.H0 * np.sqrt(
        par.OmegaM0*a**-3 + par.OmegaR0*a**-4 + par.OmegaL0)
        )

def get_τ(a):
    (τ,_)=integrate.quad(τ_itgd,0,a)
    return τ

# Find value of scale factor for which tau = tau_0.

@numba.njit
def get_H(params):
    OmegaB0, OmegaC0, OmegaG0, OmegaN0, OmegaL0, OmegaD0 = params[0:6]
    wd = numops.get_w_square(params[6 : 6+par.NC])
    OmegaD = 

    Ωd_a = Ωd_τ0 * np.exp(-3*
        integrate.cumtrapz((1+wd)/a_list,a_list, initial=0)
                      )
    H_a =  H0 *np.sqrt(Ωb0*a_list**-3 + Ωɣ0*a_list**-4 + Ωd_a + ΩΛ)*a_list
    τ_a =  integrate.cumtrapz(1/(a_list * H_a), a_list,initial=0)+τ0
    a_ =  interp1d(τ_a, a_list, kind='quadratic',fill_value='extrapolate')
    H_ = interp1d(τ_a, H_a, kind='quadratic',fill_value='extrapolate')
    Ωd_ = interp1d(τ_a, Ωd_a, kind='quadratic',fill_value='extrapolate')
    
    return par.OmegaR0

get_H(myparams)

@numba.njit
def get_TAU(params):
    return par.OmegaR0



