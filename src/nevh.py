# 2020-09-23
# JML Amoreira: jmlamoreira@gmail.com
# LJM Amoreira: amoreira@ubi.pt, ljmamoreira@gmail.com

# Rewrite of nevh. Now a callable objet is defined returning the H-gradient of
# the hamiltonian, wich can be supplied to a general ODE solver like
# numpy.integrate.solve_ivp

import numpy as np

# Hgrad: Hamiltonian grandient. 
class Hgrad():
    """
    A class to compute the hamiltonian gradient of hamiltonian funcs.

    Atributes
    ---------
    H: function H(t, [q_1,...,q_N, p_1,...,p_N])
        Hamiltonian of physical system
    N: int
        number of degrees of freedom
    dpsi: array[2N] of floats
        step sizes for partial derivatives of hamiltonian
    hparams: array[] of floats
        parameters of the hamiltonian function
    I: array[2N, 2N]
        2Nx2N identity matrix

    Methods
    -------
    __call__(t, np.array([q_1,...q_N, p_1,...p_N])
        Returns a 2N np.array with the components of the hamiltonian gradient
        [dH/dp_1,...,dH/dp_N, -dH/dq_1,...,-dH/dq_N]

    """

    def __init__(self, H, dpsi, **hparams):
        """
        Creates and inits Hgrad objects

        Parameters
        ----------
        H: function H(t, [q_1,...,q_N, p_1,...,p_N])
            Hamiltonian of physical system
        dpsi: array[2N] of floats
            step sizes for partial derivatives of hamiltonian
        hparams: array[] of floats
            parameters of the hamiltonian function

        """
        # System global parameters  
        self.H = H
        self.N = len(dpsi) // 2     # Number of degrees of freedom
        self.dpsi = dpsi            # Step sizes for differentiation
        self.hparams = hparams      # Hamiltonian parameters
        self.I = np.eye(2*self.N)   # Identity matrix

    def __call__(self, t, psi):
        # Phase space gradient 
        delta_H = np.array(
            [self.H(t, psi + self.dpsi[i]*self.I[i]/2, **self.hparams)-
             self.H(t, psi - self.dpsi[i]*self.I[i]/2, **self.hparams)
                 for i in range(2*self.N)])
        grad = delta_H / self.dpsi
        # Reordering and signs
        hgrad = np.zeros(2*self.N)
        hgrad[:self.N] = grad[self.N:]
        hgrad[self.N:] = -grad[:self.N]
        return hgrad





