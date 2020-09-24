# 2020-09-23
# JML Amoreira: jmlamoreira@gmail.com
# LJM Amoreira: amoreira@ubi.pt, ljmamoreira@gmail.com

# Rewrite of nevh. Now a callable objet is defined returning the H-gradient of
# the hamiltonian, wich can be supplied to a general ODE solver like
# numpy.integrate.solve_ivp

import numpy as np

class Hgrad():
    def __init__(self, H, N, dpsi, **hparams):
        self.H = H
        self.N = N
        self.dpsi = dpsi
        self.hparams = hparams
        self.I = np.eye(2 * N)

    def __call__(self, t, psi):
        delta_H = np.array(
            [self.H(t, psi + self.dpsi[i]*self.I[i]/2, **self.hparams)-
             self.H(t, psi - self.dpsi[i]*self.I[i]/2, **self.hparams)
                 for i in range(2*self.N)])
        grad = delta_H / self.dpsi
        hgrad = np.zeros(2*self.N)
        hgrad[:self.N] = grad[self.N:]
        hgrad[self.N:] = -grad[:self.N]
        return hgrad





