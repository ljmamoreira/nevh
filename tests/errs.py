# 2021-10-31
# amoreira@ubi.pt
# Code for checking error dependence on derivation steps

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../src")
import nevh

PI = np.pi


# Plain RK4
def srk4(t, y, dt, f, *params):
    k1 = f(t, y, *params)
    k2 = f(t + dt/2, y + k1 * dt/2, *params)
    k3 = f(t + dt/2, y + k2 * dt/2, *params)
    k4 = f(t + dt, y + k3 * dt, *params)
    return ((k1 + 2*k2+ 2*k3 + k4) * dt/6)


# Power law potential hamiltonian
# If exponent is negative, force is repulsive, but that's OK
def plh(t, psi, m, gamma, n):
    x, p = psi
    return p**2/(2*m) - gamma * x**n


# Power law potential Hamilton's eqs rhs
def pleqrhs(t, psi, m, gamma, n):
    x, p = psi
    return np.array([p/m, n*gamma*x**(n-1)])


# Compute errors for given n, given dx
def err_n(n, dx):
    m = 1.0
    gamma = 1.0
    psi_i = np.array([1.0, 0.0])
    t = 0.0
    dt = 0.05
    dpsi_s = srk4(t, psi_i, dt, pleqrhs, m, gamma, n)

    dqp = np.array([dx, 0.1], dtype=np.float32)
    hgrd = nevh.HGrad(plh, dqp, m=m, gamma=gamma, n=n)
    dpsi_n = srk4(t, psi_i, dt, hgrd)
    err = np.linalg.norm(dpsi_n - dpsi_s)
    return err


if __name__ == "__main__":

    dxs = [1e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 
           7e-6, 8e-6, 9e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    ns = [2,3,4,5]
    ls = [(),(3,3),(6,3),(9,3)]
    dxs = np.logspace(-8,-1,400)
    for n in ns:
        errs = [] #np.zeros((len(ns),len(dxs)))
        for dx in dxs:
            err = err_n(n, dx)
            #print(err)
            errs.append(err)
        #print(errs)
        plt.loglog(dxs, errs, 'k',linewidth=0.8, dashes=ls[n-2], label=r"$n="+str(n)+r"$")
    plt.xlabel(r"$\delta\nu$")
    plt.ylabel(r"$\|\epsilon\|$")
    plt.legend()
    plt.savefig("errs.png", bbox_inches='tight',dpi=300);
    plt.show()
