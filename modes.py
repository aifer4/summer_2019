"""
Perform mode evolution using 3-fluid (Seljak's 2-fluid + neutrinos) and GDM (Generalized Dark Matter).

Interface:

"""
import numpy as np
import par
import numba

# Helper functions:
# Below are implementations of the trapezoidal rule and the forward-difference
# derivative, in numba compiled functions. These are necessary because numba
# won't compile functions that use scipy.cumtrapz or numpy.gradient.
@numba.njit
def trapz(x,f):
    N = len(f)
    F = np.zeros(N)
    F[0] = 0
    for i in range(1, N):
        F[i] = F[i-1] + (x[i]-x[i-1])*(f[i]+f[i-1])/2
    return F

@numba.njit
def deriv(x,f):
    df = np.zeros(len(f))
    df[1:] = np.diff(f)/np.diff(x)
    df[0] = df[1]
    return df


def DY_3fld(i, Y, A, K, H):
    dY = np.zeros((7, len(K)))
    Phi = Y[0, :]
    deltaG = Y[1, :]
    vG = Y[2, :]
    deltaC = Y[3, :]
    vC = Y[4, :]
    deltaN = Y[5, :]
    vN = Y[6, :]

    Hi = H[i]
    H0 = par.H0
    ai = A[i]
    ybi = 1.68*ai*par.OmegaB0/par.OmegaR0
  
    OmegaBi = par.OmegaB0 * ai**-3.
    OmegaCi = par.OmegaC0 * ai**-3.
    OmegaGi = par.OmegaG0 * ai**-4.
    OmegaNi = par.OmegaN0 * ai**-4
    

    # compute the derivatives of the perturbations.
    DPhi = -Hi*Phi + (3/2.*H0**2.*ai**2/K) *\
        (4./3.*(OmegaGi*vG + OmegaNi*vN) + OmegaCi*vC + OmegaBi*vG)

    DdeltaG = -4./3.*K*vG + 4*DPhi
    DvG = (-Hi * ybi*vG + K*deltaG/3)/(
        4./3. + ybi) + K*Phi

    DdeltaC = -K*vC + 3*DPhi
    DvC = -Hi*vC + K*Phi

    DdeltaN = -4./3.*K*vN + 4*DPhi
    DvN = K*deltaN/4 + K*Phi

    DdeltaN = -4./3.*K*vN + 4*DPhi
    DvN = K*deltaN/4. + K*Phi

    dY[0, :] = DPhi
    dY[1, :] = DdeltaG
    dY[2, :] = DvG
    dY[3, :] = DdeltaC
    dY[4, :] = DvC
    dY[5, :] = DdeltaN
    dY[6, :] = DvN

    return dY

@numba.njit
def DY_2fld(i, Y, A, K, H, wD, DwD, cs2D):
    dY = np.zeros((5, len(K)))
    Phi = Y[0, :]
    deltaG = Y[1, :]
    vG = Y[2, :]
    deltaD = Y[3, :]
    vD = Y[4, :]

    # get background quantities for the current
    # time step.
    Hi = H[i]
    H0 = par.H0
    ai = A[i]
    ybi = 1.68*ai*par.OmegaB0/par.OmegaR0
    wDi = wD[i]
    DwDi = DwD[i]
    cs2Di = cs2D[:,i]
    OmegaBi = par.OmegaB0 * ai**-3.
    OmegaCi = par.OmegaC0 * ai**-3.
    OmegaGi = par.OmegaG0 * ai**-4.
    OmegaNi = par.OmegaN0 * ai**-4
    

    OmegaDi = OmegaNi + OmegaCi

    # compute the derivatives of the perturbations.
    DPhi = -Hi*Phi + (3/2.*H0**2.*ai**2/K) *\
        (4./3.*(OmegaGi*vG) + OmegaBi*vG +(1+wDi)*OmegaDi*vD)

    DdeltaG = -4./3.*K*vG + 4*DPhi
    DvG = (-Hi * ybi*vG + K*deltaG/3)/(
        4./3. + ybi) + K*Phi

    DdeltaD = -(1+wDi)*(K*vD-3*DPhi) - 3*Hi*(cs2Di-wDi)*deltaD
    DvD = -Hi*(1-3*wDi)*vD - vD*DwDi/(1+wDi) + K*deltaD*cs2Di/(1+wDi) + K*Phi

    dY[0, :] = DPhi
    dY[1, :] = DdeltaG
    dY[2, :] = DvG
    dY[3, :] = DdeltaD
    dY[4, :] = DvD

    return dY

def solve_3fld(A, K):
    N = len(A)
    H =  A * par.H0 * np.sqrt(par.OmegaM0*A**-3 + par.OmegaR0*A**-4 +  par.OmegaL0 ) 
    TAU =  par.tau0 + trapz(A, 1/(A * H))
    
    # set initial conditions
    y0 = par.a0/par.a_eq
    Phi0 = np.ones(len(K))
    deltaG0 = -2*Phi0*(1 + 3*y0/16)
    vG0 = -K/(H[0]) * (deltaG0/4 + (2*K**2 * (1 + y0)*Phi0) /
                              (9*(H[0])**2 * (4./3. + y0)))
    deltaC0 = .75 * deltaG0
    vC0 = vG0
    deltaN0 = deltaG0
    vN0 = vG0

    Y = np.zeros((N//2, 7, len(K)))
    Y[0, :, :] = np.array([Phi0, deltaG0, vG0, deltaC0, vC0, deltaN0, vN0])
    # RK4 implementation
    for i in range(N//2-1):
        ss = TAU[2*i+2] - TAU[2*i]
        k1 = ss*DY_3fld(2*i, Y[i, :, :], A,K, H)
        k2 = ss*DY_3fld(2*i+1, Y[i, :, :]+k1/2, A,K, H)
        k3 = ss*DY_3fld(2*i+1, Y[i, :, :]+k2/2, A, K,H)
        k4 = ss*DY_3fld(2*i+2, Y[i, :, :]+k3, A,K, H)

        Y[i+1, :, :] = Y[i, :, :] + k1/6 + k2/3 + k3/3 + k4/6
    return Y

@numba.njit
def solve_2fld(A, K, wD, cs2D, deltaD0, vD0):
     
    N = len(A)
    H =  A * par.H0 * np.sqrt(par.OmegaM0*A**-3 + par.OmegaR0*A**-4 +  par.OmegaL0 ) 
    TAU =  par.tau0 + trapz(A, 1/(A * H))
    DwD = deriv(TAU, wD)
    
    # set initial conditions
    y0 = par.a0/par.a_eq
    Phi0 = np.ones(len(K))
    deltaG0 = -2*Phi0*(1 + 3*y0/16)
    vG0 = -K/(H[0]) * (deltaG0/4 + (2*K**2 * (1 + y0)*Phi0) /
                              (9*(H[0])**2 * (4./3. + y0)))
    deltaC0 = .75 * deltaG0
    vC0 = vG0
    deltaN0 = deltaG0
    vN0 = vG0
    
    Y = np.zeros((N//2, 5, len(K)))
    Y[0,0,:] = Phi0
    Y[0,1,:] = deltaG0
    Y[0,2,:] = vG0
    Y[0,3,:] = deltaD0
    Y[0,4,:] = vD0
    
    #Y[0, :, :] = np.array([Phi0, deltaG0, vG0, deltaD0, vD0])
    # RK4 implementation
    for i in range(N//2-1):
        ss = TAU[2*i+2] - TAU[2*i]
        k1 = ss*DY_2fld(2*i, Y[i, :, :], A, K, H, wD, DwD, cs2D)
        k2 = ss*DY_2fld(2*i+1, Y[i, :, :]+k1/2, A, K, H, wD, DwD, cs2D)
        k3 = ss*DY_2fld(2*i+1, Y[i, :, :]+k2/2, A, K, H, wD, DwD, cs2D)
        k4 = ss*DY_2fld(2*i+2, Y[i, :, :]+k3, A, K, H, wD, DwD, cs2D)

        Y[i+1, :, :] = Y[i, :, :] + k1/6 + k2/3 + k3/3 + k4/6
    return Y
    
