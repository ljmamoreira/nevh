# 2019-08-12
# JML Amoreira: jmlamoreira@gmail.com
# LJM Amoreira: amoreira@ubi.pt, ljmamoreira@gmail.com

import numpy as np

def trajectory(H, psi0, ti, tf, nsteps, dpsi, **hparams):
    """trajectory(H, s0, ti, tf, nsteps, ds, **hparams)
    Solves Hamilton's equations for hamiltonian H and initial state s0 at nsteps
    times in the range [ti, tf]. The partial derivatives are computed using
    central derivation formulas with steps ds. 
    H must be def'ed with signature H(t, s, **hparams)
    
    Parameters:
    ===========
    H: callable with signature H(t, s, **params)
    s0: array of doubles; initial state
    ti, tf: double; initial and final times
    nsteps: integer; number of times at wich the solutions is computed
    ds: array of doubles (steps for finite difference formulas)
    **hparams: other parameters of H

    Returns:
    ========
    times: array of doubles; times at wich the solutions is computed
    traj: array of doubles width shape (2*ndf, nsteps+1); state vetors at each
          time in times
    """

    dt = (tf - ti) / nsteps
    ndim = len(psi0)                        # Number of canonical vars
    ndf = int(ndim/2)                       # Number of degrees of freedom
    times = np.linspace(ti, tf, nsteps + 1) # One more to store ti
    traj = np.zeros((2*ndf, nsteps + 1))    # One more to store S_i = (qi,pi)
    I = np.eye(ndim)
    # Using an internal, inlined definition instead of grad, so that I is
    # accessible, no need to calculate its dimensions and generate it each 
    # time grad is called
    def hgrad(H, t, psi, dpsi, **hparams):
        gradient = np.array(
                [(H(t, psi + dpsi[i]*I[i]/2, **hparams)-
                  H(t, psi - dpsi[i]*I[i]/2, **hparams))/dpsi[i]
                 for i in range(ndim)])
        hgradient = np.zeros(2*ndf)
        hgradient[:ndf] = gradient[ndf:]
        hgradient[ndf:] = -gradient[:ndf]
        return hgradient
    # All set, now compute the trajectory
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




