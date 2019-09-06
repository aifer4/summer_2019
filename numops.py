"""Numerical Operations"""
import numpy as np
import par
import numba

width = par.N_T_SOLVE//par.NC
height = par.N_K_SOLVE//par.NC

@numba.njit
def get_w_square(theta_w):
    w = np.zeros(par.N_T_SOLVE)
    for i in range(par.NC):
        w[i*width:(i+1)*width] = theta_w[i]
    return w

def square_rep_1d(vals, NUM_PTS, NUM_BINS):
    width = NUM_PTS//NUM_BINS
    f = np.zeros(NUM_PTS)
    for i in range(NUM_BINS):
        f[i*width:(i+1)*width] = vals[i]
    return f

@numba.njit
def get_cs2_square(theta_cs2):
    cs2 = np.zeros((par.N_T_SOLVE, par.N_K_SOLVE))
    for i in range(par.NC**2):
        r,c = i//par.NC, i%par.NC
        cs2[r*width:(r+1)*width, c*height:(c+1)*height] = theta_cs2[i]
    return cs2

# trapezoidal rule implementation
@numba.njit
def trapz(x,f):
    N = len(f)
    F = np.zeros(N)
    F[0] = 0
    for i in range(1, N):
        F[i] = F[i-1] + (x[i]-x[i-1])*(f[i]+f[i-1])/2
    return F

# 1d derivative
@numba.njit
def deriv(x,f):
    df = np.zeros(len(f))
    df[1:] = np.diff(f)/np.diff(x)
    df[0] = df[1]
    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    w = get_w_square(np.random.randn(par.NC))
    plt.plot(w)
    plt.plot(trapz(np.linspace(0,10,par.N_T_SOLVE),w))
    plt.show()
    cs2 = get_cs2_square(np.random.randn(par.NC**2))
    plt.imshow(cs2,aspect='auto')
    plt.show()

