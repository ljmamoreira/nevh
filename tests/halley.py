import sys
import numpy as np
import matplotlib.pyplot as plt
PI = np.pi
sys.path.append("../src")
import nevh

def rk4pl(t, y, dt, f, *params):
    k1 = f(t, y, *params)
    k2 = f(t + dt/2, y + k1 * dt/2, *params)
    k3 = f(t + dt/2, y + k2 * dt/2, *params)
    k4 = f(t + dt, y + k3 * dt, *params)
    return ((k1 + 2*k2+ 2*k3 + k4) * dt/6)


# Hamiltonian (cartesian coordinates)
def Hkep(t, psi, GM, mu):
    x, y, px, py = psi
    return (px**2 + py**2)/(2*mu) - GM*mu / (x**2 + y**2)**0.5


# Hamilton's eqs rhs
def heqsrhs(t, psi, GM, mu):
    x, y, px, py = psi
    r3 = (x**2 + y**2)**1.5
    rhs = np.array([px/mu, py/mu, -GM*mu*x/r3, -GM*mu*y/r3])
    return rhs


def halley():
    # Global physical parameters
    GM = 39.426900 # AU^3/y^2
    mu = 2.2       # 10^14 kg (Halley)
    alpha = GM*mu

    # Initial state (Halley's comet data)
    xf = 35.082 #(apheliom)
    xc = 0.586  #/periheliom
    vf = (2*GM*(1/xc - 1/xf)/((xf/xc)**2-1))**0.5

    # Angular momentum and energy
    l = mu * xf * vf
    E = mu * vf**2 / 2 - alpha / xf
    # Eccentricity, semimajor axis
    eps = (1 + 2*E*l**2/(mu*alpha**2))**0.5
    a = (xc + xf) / 2
    # Orbital period
    T = (4*PI**2 * a**3/GM)**0.5

    dt = 0.1
    dpsi = np.array([0.2, 0.2, 0.1,0.1])
    gkep = nevh.Hgrad(Hkep, dpsi, GM=GM, mu=mu)

    psi_i = [xf, 0.0, 0.0, mu*vf]

    psi_fn = rk4pl(0.0, psi_i, dt, gkep)
    psi_fs = rk4pl(0.0, psi_i, dt, heqsrhs, GM, mu)
    print(psi_i)
    print(psi_fn)
    print(psi_fs)

def hgrav1d(t, psi, GM, mu):
    x, p = psi
    return p**2 / (2*mu) - GM*mu / abs(x)

def grav1deqsrhs(t, psi, GM, mu):
    x, p = psi
    return np.array([p/mu, -GM*mu / x**2])

def grav1D():
    # Global physical parameters (arbitrary units)
    GM = 1.0
    mu = 1.0

    #Initial state, numerical steps
    psi_i = [1.0, 0.0]
    dt = 0.1
    NN = 10
    errs = np.zeros(NN-1)
    for i in range(1,NN):
        dpsi = np.array([1.0/10**i,0.1])

        hgrad = nevh.Hgrad(hgrav1d, dpsi, GM=GM, mu=mu)

        psif_n = rk4pl(0.0, psi_i, dt, hgrad)
        psif_s = rk4pl(0.0, psi_i, dt, grav1deqsrhs, GM, mu)

        #print(psi_i)
        #print(psif_s)
        #print(psif_n)
        err = np.linalg.norm(psif_s-psif_n)
        print(i, err)
        errs[i-1] = err
    plt.semilogy(errs, 'x')
    plt.show()
if __name__ == "__main__":
    grav1D()
