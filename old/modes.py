import params, background
import numpy as np
import matplotlib.pyplot as plt
import numba

"""1 fluid"""

# set initial conditions
Φ0  = Φ[:,0] 
δɣ0 = δɣ[:,0] 
vɣ0 = vɣ[:,0] 
δd0 = δd[:,0] 
vd0 = vd[:,0] 

Y0 = np.array([Φ0, δɣ0, vɣ0, δd0, vd0])

@numba.jit(nopython=True)
def DY(τ, Y, wd, Dwd):
        dY = np.zeros((5,K_SOLVE))
        Φ = Y[0,:]
        δɣ = Y[1,:]
        vɣ = Y[2,:]
        δd = Y[3,:]
        vd = Y[4,:]
           
        #compute background quantities
        ℋi = 2*α*(α*τ/τr + 1)/(α**2 * (τ**2/τr) + 2*α*τ)
        ai =  a_eq*((α*τ/τr)**2 + 2*α*τ/τr)
        ybi = 1.68*ai*Ωb0/Ωr0

        Ωbi = Ωb0 * ai**-3.
        Ωɣi = Ωɣ0 * ai**-4.
        Ωdi = Ωc0 * ai**-3. + Ων0 * ai**-4.

        wdi = np.interp(τ, τ_solve, wd)
        Dwdi = np.interp(τ, τ_solve, Dwd)
        
        #interpolate cs2
        idx = np.searchsorted(τ_solve, τ) - 1
        d = (τ - τ_solve[idx]) / (τ_solve[idx + 1] - τ_solve[idx])
        cs2di = (1 - d) * cs2d[:,idx]   + cs2d[:,idx + 1] * d
        
        f = vd*Ωdi*(1+wdi) + 4./3.*Ωɣi*vɣ +  Ωbi*vɣ
        #DΦ
        dY[0,:]  = -ℋi*Φ + (3/2.*H0**2.*ai**2/k_solve)*f
        #Dδɣ
        dY[1,:] = -4./3.*k_solve*vɣ + 4*dY[0,:]
        #Dvɣ
        dY[2,:] = (-ℋi*ybi*vɣ + k_solve*δɣ/3)/(
            4./3. + ybi) + k_solve*Φ
        #Dδd
        dY[3,:] = -(1+wdi)*(k_solve*vd-3*dY[0,:]) -\
            3*ℋi*(cs2di-wdi)*δd
        # Dvd
        dY[4,:] = -ℋi*(1-3*wdi)*vd - vd*Dwdi/(1+wdi) +\
            k_solve*δd*cs2di/(1+wdi) + k_solve*Φ

        return dY