import numpy as np
from scipy import integrate

# Numerical Settings
tau0 = 1.e-6 # start time for mode evolution
tau0_pca = 1.e-5 # earliest time for which w is varied in PCA
k_low = 1  # lowest wavenumber
k_high = 1000 # highest wavenumber
NT = 2**11 # number of points used to solve the ODE
NK = 256 # number of wavenumbers for which to solve the  ODE
NK_INT = 10000 # number of wavenumbers at which to evaluate bessel functions/integrate Cl

# PCA settings
NC = 128
w_basis = 'square'
cs2_basis = 'square'

H0 = c = 1.0 # sets the units
h = 0.6688
OmegaB0 = 0.04968
OmegaC0 = 0.26754
OmegaG0 = 5.50234e-5
OmegaN0 = 3.74248e-5 

OmegaR0 = OmegaG0 + OmegaN0
OmegaM0 = OmegaB0 + OmegaC0
OmegaL0  = 1-(OmegaM0 + OmegaR0)
deltaD0 = np.zeros(NK)
vD0 = np.zeros(NK)

wD = np.zeros(NT)
cs2D = np.zeros((NT, NK))
theta_w = np.zeros(NC)
theta_cs2 = np.zeros((NC**2))
As = np.exp(3.064)/1.e10
TCMB0 = 2.72548 * 1.e6 # microK
ns = .9667
k_star = 0.05 # Mpc^-1

a0 = 1.e-6
a0_pca = 1.e-5
z_rec = 1000. * OmegaB0 ** (-0.027 / (1 + 0.11 * np.log(OmegaB0)) )
a_rec = 1./(1. + z_rec)
a_eq  = OmegaR0/OmegaM0
tau_r = 1/(np.sqrt(OmegaM0/a_rec)*H0/2)
tau_s = 0.6 * OmegaM0**.25 * OmegaB0**(-.5) * a_rec**.75 * .67**(-.5) * tau_r
alpha = np.sqrt(a_rec/a_eq)

# seljak units
xrec = (np.sqrt((alpha**2 + 1)) - 1)/alpha
def y(x): return (alpha*x)**2 + 2*alpha*x
DeltaPhi = (2-8/y(xrec) + 16*xrec/y(xrec)**3)/(10*y(xrec))

def eta_itgd(a):
    """Conformal time integrand"""
    return 1/(a**2. * H0 * 
              np.sqrt( OmegaM0/(a**3) + OmegaR0/(a**4) + OmegaL0)
             )
# compute conformal time today
#(tau0,_) = integrate.quad(eta_itgd, 0, a0)
(tau_now,_) = integrate.quad(eta_itgd, 0, 1)
(tau_rec,_) = integrate.quad(eta_itgd, 0, a_rec)


# Plot Settings
plot_params = {'legend.fontsize': 'x-large',
               'figure.figsize': (13, 7),
               'axes.labelsize': 'x-large',
               'axes.titlesize':'x-large',
               'xtick.labelsize':'x-large',
               'ytick.labelsize':'x-large',
               'font.family' :'serif',
               'font.sans-serif' :'Garamond'}



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.pie([OmegaN0, OmegaG0, OmegaB0, OmegaC0, OmegaL0])
    plt.legend([r'$\Omega_\nu$', r'$\Omega_\gamma$', r'$\Omega_b$', r'$\Omega_c$', r'$\Omega_\Lambda$' ])
    print(OmegaR0)
    plt.show()

