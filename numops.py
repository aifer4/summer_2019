"""Numerical Operations"""
import numpy as np
import par
import numba

width = par.N_SOLVE//par.NC
height = par.K_SOLVE//par.NC

@numba.njit
def get_w_square(theta_w):
    w = np.zeros(par.N_SOLVE)
    for i in range(par.NC):
        w[i*width:(i+1)*width] = theta_w[i]
    return w


@numba.njit
def get_cs2_square(theta_cs2):
    cs2 = np.zeros((par.N_SOLVE, par.K_SOLVE))
    for i in range(par.NC**2):
        r,c = i//par.NC, i%par.NC
        cs2[r*width:(r+1)*width, c*height:(c+1)*height] = theta_cs2[i]
    return cs2

@numba.njit
def trapz(x,f):
    N = len(f)
    F = np.zeros(N)
    F[0] = 0
    for i in range(1, N):
        F[i] = F[i-1] + (x[i]-x[i-1])*(f[i]+f[i-1])/2
    return F

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    w = get_w_square(np.random.randn(par.NC))
    plt.plot(w)
    plt.plot(trapz(np.linspace(0,10,par.N_SOLVE),w))
    plt.show()
    cs2 = get_cs2_square(np.random.randn(par.NC**2))
    plt.imshow(cs2,aspect='auto')
    plt.show()

