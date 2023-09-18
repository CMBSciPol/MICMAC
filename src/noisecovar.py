import numpy as np
import healpy as hp

# Note: we only consider diagonal noise covariance


def noise_covar(freqs, depth_p):
    """
    Noise covariance matrix (nfreq).
    freqs: input frequencies
    depth_p: in uK.arcmin
    """

    return (np.ones((len(freqs))) / np.radians(depth_p/60.))**2

