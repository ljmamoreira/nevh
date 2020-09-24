# 2020-09-23
# JML Amoreira: jmlamoreira@gmail.com
# LJM Amoreira: amoreira@ubi.pt, ljmamoreira@gmail.com

# Rewrite of nevh. Now a callable objet is defined returning the H-gradient of
# the hamiltonian, wich can be supplied to a general ODE solver like
# numpy.integrate.solve_ivp

import numpy as np

class Hgrad():
    def __init__(self, H, dpsi, **hparams):
        self.H = H
        self.N = len(dpsi)
        self.ndf = self.N // 2
        self.dpsi = dpsi
        self.hparams = hparams
        self.I = np.eye(self.N)

    def __call__(self, t, psi):
        delta_H = np.array(
            [self.H(t, psi + self.dpsi[i]*self.I[i]/2, **self.hparams)-
             self.H(t, psi - self.dpsi[i]*self.I[i]/2, **self.hparams)
                 for i in range(self.N)])
        grad = delta_H / self.dpsi
        hgrad = np.zeros(self.N)
        ndf = self.N // 2
        hgrad[:self.ndf] = grad[self.ndf:]
        hgrad[self.ndf:] = -grad[:self.ndf]
        return hgrad





