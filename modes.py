"""
Perform mode evolution using 3-fluid (Seljak's 2-fluid + neutrinos) and GDM (Generalized Dark Matter).

Interface:
solve_3f

solve_gdm
"""
import numpy as np
from bg import *

tau_solve = np.array([1,2,3])

def DY_3fld(i, Y):
    dY = np.zeros((7, K_SOLVE))
    Φ = Y[0, :]
    δɣ = Y[1, :]
    vɣ = Y[2, :]
    δc = Y[3, :]
    vc = Y[4, :]
    δν = Y[5, :]
    vν = Y[6, :]

    # compute background quantities for the current
    # time step.
    ℋi = 2*α*(α*τ/τr + 1)/(α**2 * (τ**2/τr) + 2*α*τ)
    ai = a_eq*((α*τ/τr)**2 + 2*α*τ/τr)
    ybi = 1.68*ai*Ωb0/Ωr0
    Ωbi = Ωb0 * ai**-3.
    Ωci = Ωc0 * ai**-3.
    Ωɣi = Ωɣ0 * ai**-4.
    Ωνi = Ων0 * ai**-4

    # compute the derivatives of the perturbations.
    DΦ = -ℋi*Φ + (3/2.*H0**2.*ai**2/k_solve) *\
        (4./3.*(Ωɣi*vɣ + Ωνi*vν) + Ωci*vc + Ωbi*vɣ)

    Dδɣ = -4./3.*k_solve*vɣ + 4*DΦ
    Dvɣ = (-ℋi * ybi*vɣ + k_solve*δɣ/3)/(
        4./3. + ybi) + k_solve*Φ

    Dδc = -k_solve*vc + 3*DΦ
    Dvc = -ℋi*vc + k_solve*Φ

    Dδν = -4./3.*k_solve*vν + 4*DΦ
    Dvν = k_solve*δν/4 + k_solve*Φ

    Dδν = -4./3.*k_solve*vν + 4*DΦ
    Dvν = k_solve*δν/4. + k_solve*Φ

    dY[0, :] = DΦ
    dY[1, :] = Dδɣ
    dY[2, :] = Dvɣ
    dY[3, :] = Dδc
    dY[4, :] = Dvc
    dY[5, :] = Dδν
    dY[6, :] = Dvν

    return dY

def solve_3f():
    Y0 = np.zeros(4)
    for i in range(3):
        print tau_solve[i]