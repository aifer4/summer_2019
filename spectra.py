import numpy as np
from scipy.special import spherical_jn
import par
import modes
from scipy import integrate

# Set the list of k values for integration.
K_INT = np.linspace(par.k_low, par.k_high, par.NK_INT)

# Set the list of l values for C_l.
l_list = np.array([i for i in range(2, 50, 1)]+[i for i in range(50, 200, 5)] +
                  [i for i in range(200, 2500, 20)])
L = len(l_list)

"""Precompute Bessel Functions"""
# check if they're already stored
def get_bessels():
    global JLK, DJLK
    JLK = np.array([
        spherical_jn(l, K_INT*(par.tau_now-par.tau_rec))
        for l in l_list])
    DJLK = np.array([
        spherical_jn(l, K_INT*(par.tau_now-par.tau_rec), derivative=True)
        for l in l_list])
    
if 'DJLK' not in globals() or 'JLK' not in globals():
    print('precomputing bessel functions...\n')
    get_bessels()
    print('bessel functions precomputed.\n')
          
k_grid = np.broadcast_to(K_INT, (len(l_list), par.NK_INT))
def get_Cl(A, K, wD, cs2D, deltaD0, vD0):
    # first compute mode evolution:
    Y = modes.solve_2fld(A, K, wD, cs2D, deltaD0, vD0)
    
    SW = Y[-1, 0, :] + Y[-1,1, :]/4
    ISW = par.DeltaPhi
    DOP = Y[-1, 2, :]

    SWsd = (SW+ISW)*np.exp(-(K*par.tau_s)**2)
    DOPsd = DOP*np.exp(-(K*par.tau_s)**2)

    # get the power spectrum
    SWfill = np.interp(K_INT, K, SW)
    DOPfill = np.interp(K_INT, K, DOP)
    Dl = SWfill*JLK + DOPfill*(DJLK-JLK/(2*K_INT*(par.tau_now-par.tau_rec)))

    T = np.exp(-2*(K_INT*par.tau_s)**2 - (.03*K_INT*par.tau_rec)**2)
    Cl_itgd = Dl**2 * T / K_INT
    Cl = integrate.trapz(k_grid, Cl_itgd)
    # this normalization makes the most sense, but it still seems to be too large
    # by a factor of two (at the first peak).
    norm = (4)*np.pi * par.As * par.TCMB0**2
    Cl_normed = l_list*(l_list + 1)*Cl * norm/(2*np.pi)
    return np.abs(Cl_normed)

def get_Cl_err():
    # Error:
    f_sky = 1.0  # fraction of sky
    l_s = 500.  # filtering scale
    theta_pix = 0.0012  # rad
    sigma_pix = 16.e-6
    wbar = 1/(0.33e-15)
    B_cl = np.exp(-l_list*(l_list + 1)/l_s**2)
    rel_err = (2/((2*l_list+1)*f_sky)) * (2000 + wbar**(-1) * B_cl**-2)**2
    return rel_err

          