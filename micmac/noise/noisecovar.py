# This file is part of MICMAC.
# Copyright (C) 2024 CNRS / SciPol developers
#
# MICMAC is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MICMAC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MICMAC. If not, see <https://www.gnu.org/licenses/>.

"""
Noise covariance matrix and useful recurring operations.
"""
import healpy as hp
import jax.numpy as jnp
import numpy as np

# Note: we only consider diagonal noise covariance

__all__ = [
    'get_noise_covar',
    'get_noise_covar_extended',
    'get_BtinvN',
    'get_BtinvNB',
    'get_inv_BtinvNB',
    'get_Wd',
    'get_Cl_noise',
    'get_Cl_noise_JAX',
    'get_freq_inv_noise_JAX',
    'get_inv_BtinvNB_c_ell',
    'get_true_Cl_noise',
]


### Objects in pixel domain
def get_noise_covar(depth_p, nside):
    """
    Noise covariance matrix (nfreq*nfreq).

    Parameters
    ----------
    depth_p: array[float] of dimensions [nfreq]
        polarization depth in uK.arcmin, one value per freq channel
    nside: int
        nside of the input maps

    Returns
    -------
    invN: array[float] of dimensions [nfreq, nfreq]
        inverse noise covariance matrix in uK^-2
    """
    invN = np.linalg.pinv(np.diag((depth_p / hp.nside2resol(nside, arcmin=True)) ** 2))

    return invN


def get_noise_covar_extended(depth_p, nside):
    """
    Noise covariance matrix (nfreq*nfreq) extended to the pixel domain.

    Parameters
    ----------
    depth_p: array[float] of dimensions [nfreq]
        polarization depth in uK.arcmin, one value per freq channel
    nside: int
        nside of the input maps

    Returns
    -------
    invN: array[float] of dimensions [nfreq, nfreq, 12*nside**2]
        inverse noise covariance matrix in uK^-2 extended to pixel domain
    """
    invN = get_noise_covar(depth_p, nside)
    # invN_extended = np.repeat(invN.ravel(order='F'), 12*nside**2).reshape((invN.shape[0],invN.shape[0],12*nside**2), order='C')
    invN_extended = np.broadcast_to(invN, (12 * nside**2, invN.shape[0], invN.shape[0])).swapaxes(0, 2)
    return invN_extended


def get_BtinvN(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.

    Parameters
    ----------
    invN: array[float] of dimensions [nfreq, nfreq, ...]
        inverse noise covariance matrix in uK^-2
    B: array[float] of dimensions [nfreq, ncomp, ...]
        mixing matrix
    jax_use: bool
        whether to use jax or not for the einsum operation

    Returns
    -------
    BtinvN: array[float] of dimensions [ncomp, nfreq, ...]
        matrix product B^T * invN
    """

    if jax_use:
        return jnp.einsum('fc...,fh...->ch...', B, invN)

    return np.einsum('fc...,fh...->ch...', B, invN)


def get_BtinvNB(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.

    Parameters
    ----------
    invN: array[float] of dimensions [nfreq, nfreq, ...]
        inverse noise covariance matrix in uK^-2
    B: array[float] of dimensions [nfreq, ncomp, ...]
        mixing matrix
    jax_use: bool
        whether to use jax or not for the einsum operation

    Returns
    -------
    BtinvNB: array[float] of dimensions [ncomp, nfreq, nfreq, ...]
        matrix product B^T * invN * B
    """
    BtinvN = get_BtinvN(invN, B, jax_use)

    if jax_use:
        return jnp.einsum('ch...,hf...->cf...', BtinvN, B)

    return np.einsum('ch...,hf...->cf...', BtinvN, B)


def get_inv_BtinvNB(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.

    Parameters
    ----------
    invN: array[float] of dimensions [nfreq, nfreq, ...]
        inverse noise covariance matrix in uK^-2
    B: array[float] of dimensions [nfreq, ncomp, ...]
        mixing matrix
    jax_use: bool
        whether to use jax or not for the einsum and inverse operations

    Returns
    -------
    invBtinvNB: array[float] of dimensions [ncomp, nfreq, nfreq, ...]
        pseudo-inverse of matrix product B^T * invN * B
    """
    BtinvNB = get_BtinvNB(invN, B, jax_use)

    if jax_use:
        return jnp.linalg.pinv(BtinvNB.swapaxes(0, -1)).swapaxes(0, -1)

    return np.linalg.pinv(BtinvNB.swapaxes(0, -1)).swapaxes(0, -1)


def get_Wd(invN, B, d, jax_use=False):
    """
    Returns W d = inv(Bt invN B) Bt invN d

    Parameters
    ----------
    invN: array[float] of dimensions [nfreq, nfreq, ...]
        inverse noise covariance matrix
    B: array[float] of dimensions [nfreq, ncomp, ...]
        mixing matrix
    d: array[float] of dimensions [nfreq, ...]
        frequency maps

    Returns
    -------
    Wd: array[float] of dimensions [ncomp, ...]
        W d = inv(Bt invN B) Bt invN d
    """
    invBtinvNB = get_inv_BtinvNB(invN, B, jax_use)

    if jax_use:
        return jnp.einsum('cg...,hg...,hf...,f...->c...', invBtinvNB, B, invN, d)

    return np.einsum('cg...,hg...,hf...,f...->c...', invBtinvNB, B, invN, d)


### Objects in harmonic domain
def get_Cl_noise(depth_p, A, lmax, jax_use=False):
    """
    Function used only in harmonic case
    A: is pixel independent MixingMatrix,
    thus if you want to get it from full MixingMatrix
    you have to select the entry correspondind to one pixel

    Parameters
    ----------
    depth_p: array[float] of dimensions [nfreq]
        polarization depth in uK.arcmin, one value per freq channel
    A: array[float] of dimensions [ncomp, nfreq]
        mixing matrix
    lmax: int
        maximum ell for the spectrum

    Returns
    -------
    inv_AtNA: array[float] of dimensions [ncomp, ncomp, lmax+1]
        inverse of At N^-1 A in harmonic domain
    """

    if jax_use:
        return get_Cl_noise_JAX(depth_p, A, lmax)

    assert len(np.shape(A)) == 2
    bl = np.ones((len(depth_p), lmax + 1))

    nl = (bl / np.radians(depth_p / 60.0)[:, np.newaxis]) ** 2
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl, A)
    inv_AtNA = np.linalg.pinv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)


def get_Cl_noise_JAX(depth_p, A, lmax):
    """
    Note: Function for testing purposes

    Function used only in harmonic case
    A: is pixel independent MixingMatrix,
    thus if you want to get it from full MixingMatrix
    you have to select the entry correspondind to one pixel

    Parameters
    ----------
    depth_p: array[float] of dimensions [nfreq]
        polarization depth in uK.arcmin, one value per freq channel
    A: array[float] of dimensions [ncomp, nfreq]
        mixing matrix
    lmax: int
        maximum ell for the spectrum

    Returns
    -------
    inv_AtNA: array[float] of dimensions [ncomp, ncomp, lmax+1]
        inverse of At N^-1 A in harmonic domain
    """
    assert len(np.shape(A)) == 2
    bl = jnp.ones((jnp.size(depth_p), lmax + 1))

    nl = (bl / jnp.radians(depth_p / 60.0)[:, jnp.newaxis]) ** 2
    AtNA = jnp.einsum('fi, fl, fj -> lij', A, nl, A)
    inv_AtNA = jnp.linalg.pinv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)


def get_freq_inv_noise_JAX(depth_p, lmax):
    return jnp.einsum('fg,l->fgl', jnp.diag(1 / jnp.radians(depth_p / 60.0), jnp.ones(lmax + 1)))


def get_inv_BtinvNB_c_ell(freq_inv_noise, mixing_matrix):
    """
    Function used only in harmonic case
    mixing_matrix: is pixel independent MixingMatrix,
    thus if you want to get it from full MixingMatrix
    you have to select the entry correspondind to one pixel

    Parameters
    ----------
    freq_inv_noise: array[float] of dimensions [nfreq, nfreq, lmax+1]
        inverse noise covariance matrix in uK^-2
    mixing_matrix: array[float] of dimensions [ncomp, nfreq]
        mixing matrix

    Returns
    -------
    BtinvNB: array[float] of dimensions [lmax+1, ncomp, ncomp]
        pseudo-inverse of matrix product B^T * invN * B
    """
    BtinvNB = jnp.einsum('fc,fgl,gk->lck', mixing_matrix, freq_inv_noise, mixing_matrix)
    return jnp.linalg.pinv(BtinvNB).swapaxes(-3, -1)


def get_true_Cl_noise(depth_p, lmax):
    """
    Function used only in harmonic case
    Returns the inverse noise power spectrum

    Parameters
    ----------
    depth_p: array[float] of dimensions [nfreq]
        polarization depth in uK.arcmin, one value per freq channel
    lmax: int
        maximum ell for the spectrum

    Returns
    -------
    Cl_noise: array[float] of dimensions [nfreq, nfreq, lmax+1]
        Cl noise
    """
    bl = jnp.ones((jnp.size(depth_p), lmax + 1))

    nl = (bl / jnp.radians(depth_p / 60.0)[:, jnp.newaxis]) ** 2
    return jnp.einsum('fl,fk->fkl', nl, jnp.eye(jnp.size(depth_p)))
