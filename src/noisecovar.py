"""
Noise covariance matrix and useful recurring operations.
"""
import numpy as np
import healpy as hp

# Note: we only consider diagonal noise covariance


def get_noise_covar(depth_p, nside):
    """
    Noise covariance matrix (nfreq*nfreq).
    depth_p: in uK.arcmin, one value per freq channel
    nside: nside of the input maps
    """
    invN = np.linalg.inv(np.diag((depth_p / hp.nside2resol(nside, arcmin=True))**2))
    
    return invN


def get_inv_BtinvNB(invN, B):
    """
    B can be full Mixing Matrix, 
    or just the cmb part, 
    or just the fgs part.
    """
    BtinvNB = np.einsum('fc,fh,hg->cg', B, invN, B)
    invBtinvNB = np.linalg.inv(BtinvNB)
    
    return invBtinvNB