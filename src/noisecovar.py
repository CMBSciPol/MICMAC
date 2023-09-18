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


def get_BtinvN(invN, B):
    """
    B can be full Mixing Matrix, 
    or just the cmb part, 
    or just the fgs part.
    """
    BtinvN = np.einsum('fc,fh->ch', B, invN)
    
    return BtinvN



## Choose if we want to keep these ones
## (this could be done directly in the code)
def select_cmb_EtX(X, ncomp):
    """
    It acts on object w ncomp as external dimension 
    and it selects only cmb.
    """
    # check that the ext dim is indeed ncomp
    assert X.shape[0] == ncomp
    EtX = X[0, ...]
    
    return EtX


def select_cmb_EtXE(X, ncomp):
    """
    It acts on object w (ncomp,ncomp) as external dimensions
    and it selects only cmb.
    """
    # check that the ext dims are indeed (ncomp,ncomp)
    assert X.shape[0] == ncomp
    assert X.shape[1] == ncomp
    EtXE = X[0, 0, ...]
    
    return EtXE
