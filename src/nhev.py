import numpy as np
from copy import deepcopy


def grad(H, t, psi, dpsi, **hparams):
    """grad(H, t, psi, dpsi, **hparams)
    Given function H(t, psi, **hparams), return numerical estimate of gradient
    at t, psi, using step vector dpsi.
    The gradient is computed using central differences. The returned gradient
    has shape equal to psi.shape. 

    Parameters:
    ===========
    H: callable with signature H(t, psi, *params)
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


def hgrad(H, t, psi, dpsi, **hparams):
    """hgrad(H, t, psi, dpsi, **params)
    Given function H(t, psi, **hparams), return numerical estimate of
    "Hamilton's gradient", (dH/dpsi[1], -dH/dpsi[0])
    The partial derivatives are computed using grad. hgrad has shape equal to
    psi's.

    Parameters:
    ===========
    H: callable with signature H(t, psi, *params)
    t: double
    psi: array of doubles storing qs and ps
    dpsi: array of doubles (steps for finite difference formulas)
    **hparams: other parameters of H

    Returns:
    ========
    gradient: ndarray

    Examples:
    =========
    >>> hgrad(lambda t,x: x[0]+x[1]**2, 0, [1,1] ,[0.1,0.1])
    array([ 2., -1.])
    """
    ndf = int(len(gradient)/2)
    i = np.eye(ndf)
    z = np.zeros((ndf,ndf))
    hswap = np.vstack(
            (np.hstack((z,i)),
             np.hstack((-i,z)))
            )

    gradient = np.array(
            [(H(t, psi + dpsi[i]*identity[i]/2, **hparams)-
              H(t, psi - dpsi[i]*identity[i]/2, **hparams))/dpsi[i]
             for i in range(n)])
    hgradient = np.zeros(2*ndf)
    hgradient[:ndf] = gradient[ndf:]
    hgradient[ndf:] = -gradient[:ndf]
    return hgradient


# Signature for hamiltonian functions: H(t, psi, **hparamss)
# q, p -> arrays with dim = ndf (number of degrees of fereedom)
# t -> time
# **hparams -> Other, problem-specific parameters in H
#               (masses, elastic constants, g, etc)
# dp, dq -> deltas for partial derivatives
# nsteps -> time resolution for solution: number of instants where q,p are
#               computed
# trajectory() returns two arrays. The first (shape=(nsteps+1,)) is the list
#               of instants when (p,q) is computed. The second
#               (shape=(2*ndf,nsteps+1)) stores the actual values of q and p

def trajectory(H, psi0, ti, tf, nsteps, dpsi, **hparams):
    # nsteps is the number of snapshots computed. First is at ti+dt, last at tf.
    dt = (tf - ti) / nsteps
    ndf = int(len(psi0)/2)       # Number of degrees of freedom
    times = np.linspace(ti, tf, nsteps + 1) # One more to store ti
    traj = np.zeros((2*ndf, nsteps + 1))      # One more to store S_i = (qi,pi)
    
    traj[:, 0] = psi0
    psin = np.copy(psi0)
    for t_index, t in enumerate(times[:-1]):
        psic = np.copy(psin)
        k1 = dt * hgrad(H, t, psic, dpsi, **hparams)
        k2 = dt * hgrad(H, t+dt/2, psic+k1/2, dpsi, **hparams)
        k3 = dt * hgrad(H, t+dt/2, psic+k2/2, dpsi, **hparams)
        k4 = dt * hgrad(H, t+dt,   psic+k3  , dpsi, **hparams)
        psin = psic + (k1 + 2*k2 + 2*k3 + k4) / 6
        traj[:,t_index+1] = np.copy(psin)
    return times, traj

