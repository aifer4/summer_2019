"""Evaluate background quantities."""

import par
from scipy import integrate
from scipy.optimize import root
import numpy as np
import numba
import numops
      
myparams = par.MLparams


def tau_itgd_early(a):
    # conformal time integrand valid for tau < tau0_pca (when wd = wd_ML).
    return  1/(a**2* par.H0 * np.sqrt(
        par.OmegaM0*a**-3 + par.OmegaR0*a**-4 + par.OmegaL0)
        )

@np.vectorize
def get_tau_early(a):
    (tau,_)=integrate.quad(tau_itgd_early,0,a)
    return tau

# Find value of scale factor for which tau = tau_0.
a0 = root(lambda a: get_tau_early(a) - par.tau0, par.tau0).x[0]
A = np.linspace(a0, par.a_rec, par.N_T_SOLVE)

@numba.njit 
def get_OmegaD(params):
    wd = numops.get_w_square(params[6:6+par.NC])
    OmegaD = np.zeros(par.N_T_SOLVE)
    s = -3*numops.trapz(A, (1+wd)/A)
    OmegaD = par.OmegaD_tau0 * np.exp(s)
    return OmegaD

@numba.njit
def get_H(params):
    H = np.zeros(par.N_T_SOLVE)
    OmegaB0, OmegaC0, OmegaG0, OmegaN0, OmegaL0, OmegaD_tau0 = params[0:6]
    OmegaD = get_OmegaD(params)
    OmegaM0 = OmegaB0 + OmegaC0
    OmegaR0 = OmegaG0 + OmegaN0
    H = A * par.H0 * np.sqrt(OmegaM0*A**-3 + OmegaR0*A**-4 +  OmegaL0 + OmegaD)
    return H

@numba.njit
def get_TAU(params):
    H = get_H(params)
    TAU =  par.tau0 + numops.trapz(A, 1/(A * H))
    return TAU

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    H = get_H(myparams)
    OmD = get_OmegaD(myparams)
    tau = get_TAU(myparams)
    plt.plot(A,H)
    plt.show()
    plt.plot(A,tau)
    plt.show()

