import numpy as np

H0 = c = 1.0 # sets the units

# specified parameters
h = 0.6688
OmegaB0 = .0223 / h**2.
OmegaL0 = 0.679
OmegaC0 = 0.1206 / h**2.
OmegaN0 = 1.68e-5 / h**2.
OmegaD0 = 0.

As = np.exp(3.064)/1.e10
TCMB0 = 2.72548 * 1.e6 # μK

# derived parameters
OmegaM0 = OmegaB0 + OmegaC0
OmegaR0 = 1 - (OmegaM0 + OmegaL0 + OmegaD0) 
OmegaG0 = OmegaR0 - OmegaN0
OmegaB0 = OmegaM0 - OmegaC0
z_rec = 1000. * OmegaB0 ** (-0.027 / (1 + 0.11 * np.log(OmegaB0)) )
a_rec = 1./(1. + z_rec)
a_eq  = OmegaR0/OmegaM0

tau_r = 1/(np.sqrt(OmegaM0/a_rec)*H0/2)
tau_s = 0.6 * OmegaM0**.25 * OmegaB0**(-.5) * a_rec**.75 * .67**(-.5) * tau_r
alpha = np.sqrt(a_rec/a_eq)

params = np.array([OmegaB0, OmegaC0, OmegaG0, OmegaN0, OmegaL0, OmegaD0])


# Numerical Settings
tau0 = 1.e-7 # start time for mode evolution
tau0_pca = 1.e-5 # earliest time for which w is varied in PCA
k_low = 1  # lowest wavenumber
k_high = 1000 # highest wavenumber
N_SOLVE = 2**12 # number of points used to solve the ODE
K_SOLVE = 256 # number of wavenumbers for which to solve the  ODE
K_INT = 10000 # number of wavenumbers at which to evaluate bessel functions/integrate Cl

# PCA settings
NC = 16
w_basis = 'square'
cs2_basis = 'square'

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.pie([OmegaN0, OmegaG0, OmegaB0, OmegaC0, OmegaL0])
    plt.legend([r'$\Omega_\nu$', r'$\Omega_\gamma$', r'$\Omega_b$', r'$\Omega_c$', r'$\Omega_\Lambda$' ])
    print(OmegaN0, OmegaG0)
    plt.show()

