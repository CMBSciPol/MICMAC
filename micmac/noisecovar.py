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


### Objects in pixel domain
def get_noise_covar(depth_p, nside):
    """
    Noise covariance matrix (nfreq*nfreq).
    depth_p: in uK.arcmin, one value per freq channel
    nside: nside of the input maps
    """
    invN = np.linalg.inv(np.diag((depth_p / hp.nside2resol(nside, arcmin=True)) ** 2))

    return invN


def get_noise_covar_extended(depth_p, nside):
    invN = get_noise_covar(depth_p, nside)
    # invN_extended = np.repeat(invN.ravel(order='F'), 12*nside**2).reshape((invN.shape[0],invN.shape[0],12*nside**2), order='C')
    invN_extended = np.broadcast_to(invN, (12 * nside**2, invN.shape[0], invN.shape[0])).swapaxes(0, 2)
    return invN_extended


def get_BtinvN(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.
    """

    if jax_use:
        return jnp.einsum('fc...,fh...->ch...', B, invN)

    return np.einsum('fc...,fh...->ch...', B, invN)


def get_BtinvNB(invN, B, jax_use=False):
    """
    B can be full Mixing Matrix,
    or just the cmb part,
    or just the fgs part.
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
    """
    BtinvNB = get_BtinvNB(invN, B, jax_use)

    if jax_use:
        return jnp.linalg.pinv(BtinvNB.swapaxes(0, -1)).swapaxes(0, -1)

    return np.linalg.pinv(BtinvNB.swapaxes(0, -1)).swapaxes(0, -1)


def get_Wd(invN, B, d, jax_use=False):
    """
    invN: inverse noise covar matrix
    B: mixing matrix
    d: frequency maps
    returns: W d = inv(Bt invN B) Bt invN d
    """
    invBtinvNB = get_inv_BtinvNB(invN, B, jax_use)

    if jax_use:
        return jnp.einsum('cg...,hg...,hf...,f...->c...', invBtinvNB, B, invN, d)

    return np.einsum('cg...,hg...,hf...,f...->c...', invBtinvNB, B, invN, d)


### Objects in harmonic domain
def get_Cl_noise(depth_p, A, lmax):
    """
    Function used only in harmonic case
    A: is pixel independent MixingMatrix,
    thus if you want to get it from full MixingMatrix
    you have to select the entry correspondind to one pixel
    """
    assert len(np.shape(A)) == 2
    bl = np.ones((len(depth_p), lmax + 1))

    nl = (bl / np.radians(depth_p / 60.0)[:, np.newaxis]) ** 2
    AtNA = np.einsum('fi, fl, fj -> lij', A, nl, A)
    inv_AtNA = np.linalg.inv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)


def get_Cl_noise_JAX(depth_p, A, lmax):
    """
    Function used only in harmonic case
    A: is pixel independent MixingMatrix,
    thus if you want to get it from full MixingMatrix
    you have to select the entry correspondind to one pixel
    """
    assert len(np.shape(A)) == 2
    bl = jnp.ones((jnp.size(depth_p), lmax + 1))

    nl = (bl / jnp.radians(depth_p / 60.0)[:, jnp.newaxis]) ** 2
    AtNA = jnp.einsum('fi, fl, fj -> lij', A, nl, A)
    inv_AtNA = jnp.linalg.inv(AtNA)
    return inv_AtNA.swapaxes(-3, -1)


def get_freq_inv_noise_JAX(depth_p, lmax):
    return jnp.einsum('fg,l->fgl', jnp.diag(1 / jnp.radians(depth_p / 60.0), jnp.ones(lmax + 1)))


def get_inv_BtinvNB_c_ell(freq_inv_noise, mixing_matrix):
    """
    Function used only in harmonic case
    mixing_matrix: is pixel independent MixingMatrix,
    thus if you want to get it from full MixingMatrix
    you have to select the entry correspondind to one pixel
    """
    BtinvNB = jnp.einsum('fc,fgl,gk->lck', mixing_matrix, freq_inv_noise, mixing_matrix)
    return jnp.linalg.pinv(BtinvNB).swapaxes(-3, -1)


def get_true_Cl_noise(depth_p, lmax):
    bl = jnp.ones((jnp.size(depth_p), lmax + 1))

    nl = (bl / jnp.radians(depth_p / 60.0)[:, jnp.newaxis]) ** 2
    return jnp.einsum('fl,fk->fkl', nl, jnp.eye(jnp.size(depth_p)))
