"""Evaluate background quantities."""

import par
from scipy import integrate
from scipy.optimize import root
import numpy as np
import numba
import numops


def get_OmegaD(A, params):
    wd = params['wD']
    OmegaD = np.zeros(par.N_T_SOLVE)
    s = -3*numops.trapz(A, (1+wd)/A)
    OmegaD = params['OmegaD_tau0'] * np.exp(s)
    return OmegaD

def get_H(A, params):
    H = np.zeros(par.N_T_SOLVE)
    # OmegaD = get_OmegaD(params)
    OmegaM0 = params['OmegaB0'] + params['OmegaC0']
    OmegaR0 = params['OmegaG0'] + params['OmegaN0']
    H = A * par.H0 * np.sqrt(OmegaM0*A**-3 + OmegaR0*A**-4 +  params['OmegaL'] ) 
    return H

def get_TAU(A, params):
    H = get_H(A,params)
    TAU =  par.tau0 + numops.trapz(A, 1/(A * H))
    return TAU

