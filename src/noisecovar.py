"""
Noise covariance matrix and useful recurring operations.
"""
import numpy as np
import healpy as hp
import jax.numpy as jnp

# Note: we only consider diagonal noise covariance


def get_noise_covar(depth_p, nside):
    """
    Noise covariance matrix (nfreq*nfreq).
    depth_p: in uK.arcmin, one value per freq channel
    nside: nside of the input maps
    """
    invN = np.linalg.inv(np.diag((depth_p / hp.nside2resol(nside, arcmin=True)) ** 2))

    return invN

def get_Cl_noise(depth_p, A, lmax):
    bl = np.ones((len(depth_p), lmax+1))

    nl = (bl / np.radians(depth_p/60.)[:, np.newaxis])**2
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl, A)
    inv_AtNA = np.linalg.inv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)

def get_Cl_noise_from_invBtinvNB(invBtinvNB, nstokes, nside, lmax):
    """
        Return cl noise from invBtinvNB if invBtinvNB is not multi-resolution
    """
    number_correlations = int(jnp.ceil(nstokes**2/2) + jnp.floor(nstokes/2))
    full_spectra = jnp.zeros((number_correlations,lmax+1))
    full_spectra = full_spectra.at[:nstokes,:].set(invBtinvNB*hp.nside2resol(nside)**2)
    return full_spectra

def get_BtinvN(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.
    """

    if jax_use:
        return jnp.einsum("fc,fh->ch", B, invN)

    return np.einsum("fc,fh->ch", B, invN)


def get_BtinvNB(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.
    """
    BtinvN = get_BtinvN(invN, B, jax_use)

    if jax_use:
        return jnp.einsum("ch,hf->cf", BtinvN, B)

    return np.einsum("ch,hf->cf", BtinvN, B)


def get_inv_BtinvNB(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.
    """
    BtinvNB = get_BtinvNB(invN, B, jax_use)

    if jax_use:
        return jnp.linalg.inv(BtinvNB)

    return np.linalg.inv(BtinvNB)


def get_Wd(invN, B, d, jax_use=False):
    """
    invN: inverse noise covar matrix
    B: mixing matrix
    d: frequency maps
    returns: W d = inv(Bt invN B) Bt invN d
    """
    invBtinvNB = get_inv_BtinvNB(invN, B, jax_use)

    if jax_use:
        return jnp.einsum("cg,hg,hf,fsp->csp", invBtinvNB, B, invN, d)

    return np.einsum("cg,hg,hf,fsp->csp", invBtinvNB, B, invN, d)


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
