def hgrad(H, t, psi, dpsi, **hparams):
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

