# 2019-08-12
# JML Amoreira: jmlamoreira@gmail.com
# LJM Amoreira: amoreira@ubi.pt, ljmamoreira@gmail.com

import numpy as np


def grad(H, t, psi, dpsi, **hparams):
    """grad(H, t, psi, dpsi, **hparams)
    Given function H(t, psi, **hparams), return numerical estimate of gradient
    at t, psi, using step vector dpsi.
    The gradient is computed using central differences. The returned gradient
    has shape equal to psi.shape. 

    Parameters:
    ===========
    H: callable with signature H(t, psi, **params)
    t: double
    psi: array of doubles storing qs and ps
    dpsi: array of doubles (steps for finite difference formulas)
    **hparams: other parameters of H

    Returns:
    ========
    gradient: ndarray
    
    Examples:
    =========
    >>> grad(lambda t,x: x[0]+x[1]**2, 0, [1,1] ,[0.1,0.1])
    array([1., 2.])
    """
    n = len(psi)
    identity = np.eye(n)
    return np.array(
            [(H(t, psi + dpsi[i]*identity[i]/2, **hparams)-
              H(t, psi - dpsi[i]*identity[i]/2, **hparams))/dpsi[i]
             for i in range(n)])


