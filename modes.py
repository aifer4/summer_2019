"""
Perform mode evolution using 3-fluid (Seljak's 2-fluid + neutrinos) and GDM (Generalized Dark Matter).

Interface:
solve_3f

solve_gdm
"""
import numpy as np
import bg
import par
import numba

# coordinate arrays
TAU = bg.get_TAU(par.MLparams)
K = np.linspace(par.k_low, par.k_high, par.N_K_SOLVE)
A = bg.A

# initialize background quantities
H = bg.get_H(par.MLparams)
OmegaD = bg.get_OmegaD(par.MLparams)
[OmegaB0, OmegaC0, OmegaG0, OmegaN0, OmegaL0, OmegaD_tau0] = par.MLparams[0:6]
a_eq = (OmegaG0 + OmegaN0)/(OmegaB0 + OmegaC0)

@numba.njit(cache=True)
def DY_3fld(i, Y):
    dY = np.zeros((7, par.N_K_SOLVE))
    phi = Y[0, :]
    delta_g = Y[1, :]
    v_g = Y[2, :]
    delta_c = Y[3, :]
    vc = Y[4, :]
    delta_n = Y[5, :]
    v_n = Y[6, :]

    ai = A[i]
    Hi = H[i]
    ybi = 1.68*ai*OmegaB0/(OmegaG0 + OmegaN0)
    OmegaBi = OmegaB0 * ai**-3.
    OmegaCi = OmegaC0 * ai**-3.
    OmegaGi = OmegaG0 * ai**-4. 
    OmegaNi = OmegaN0 * ai**-4

    # compute the derivatives of the perturbations.
    Dphi = -Hi*phi + (3/2.*par.H0**2.*ai**2/K) *\
        (4./3.*(OmegaGi*v_g + OmegaNi*v_n) + OmegaCi*vc + OmegaBi*v_g)

    Ddelta_g = -4./3.*K*v_g + 4*Dphi
    Dv_g = (-Hi * ybi*v_g + K*delta_g/3)/(
        4./3. + ybi) + K*phi

    Ddelta_c = -(K*vc) + 3*Dphi
    Dvc = -Hi*vc + K*phi

    Ddelta_n = -4./3.*K*v_n + 4*Dphi
    Dv_n = K*delta_n/4 + K*phi

    Ddelta_n = -4./3.*K*v_n + 4*Dphi
    Dv_n = K*delta_n/4. + K*phi

    dY[0, :] = Dphi
    dY[1, :] = Ddelta_g
    dY[2, :] = Dv_g
    dY[3, :] = Ddelta_c
    dY[4, :] = Dvc
    dY[5, :] = Ddelta_n
    dY[6, :] = Dv_n

    return dY

@numba.njit(cache=True)
def solve_3fld(params):
    # get background qualities
    H = bg.get_H(params)
    OmegaD = bg.get_OmegaD(params)
    [OmegaB0, OmegaC0, OmegaG0, OmegaN0, OmegaL0, OmegaD_tau0] = params[0:6]
    a_eq = (OmegaG0 + OmegaN0)/(OmegaB0 + OmegaC0)

    # output array is half the length because RK4 throws out every other time step.
    Y = np.zeros((par.N_T_SOLVE//2, 7, par.N_K_SOLVE))

    # set initial conditions
    phi0 = np.ones(par.N_K_SOLVE)
    delta_g0 = -2*phi0*(1 + 3*(A[0]/a_eq))/16
    v_g0 = -K/H[0] * (delta_g0/4 + (2*K**2 * (1 + (A[0]/a_eq)))*phi0) /\
                            (9*H[0]**2 * (4./3. + (A[0]/a_eq)))
    delta_c0 = .75 * delta_g0
    v_c0 = v_g0
    delta_n0 = delta_g0
    v_n0 = v_g0
    Y0 = np.stack((phi0, delta_g0, v_g0, delta_c0, v_c0, delta_n0, v_n0))
    # set initial conditions:
    Y[0,:,:] = Y0

    for i in range(par.N_T_SOLVE//2-1):
        h = TAU[i+2] - TAU[i]
        k1 = h*DY_3fld(2*i,Y[i,:,:])
        k2 = h*DY_3fld(2*i+1,Y[i,:,:]+k1/2)
        k3 = h*DY_3fld(2*i+1,Y[i,:,:]+k2/2)
        k4 = h*DY_3fld(2*i+2,Y[i,:,:]+k3)
        Y[i+1,:,:] = Y[i,:,:] + k1/6 + k2/3 + k3/3 + k4/6
    return Y


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Y = solve_3fld(par.MLparams)
    print(np.shape(Y))
    plt.plot(Y[:,0, 200])
    plt.show()
    
    