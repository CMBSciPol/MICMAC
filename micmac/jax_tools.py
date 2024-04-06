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

from functools import partial

import chex as chx
import healpy as hp
import jax
import jax.lax as jlax
import jax.numpy as jnp
import numpy as np


def get_reduced_matrix_from_c_ell_jax(c_ells_input):
    """
    Returns the input spectra in the format [lmax+1-lmin, nstokes, nstokes]

    Expect c_ells_input to be sorted as TT, EE, BB, TE, TB, EB if 6 spectra are given
    or EE, BB, EB if 3 spectra are given
    or TT if 1 spectrum is given

    Generate covariance matrix from c_ells assuming it's block diagonal,
    in the "reduced" (prefix red) format, i.e. : [ell, nstokes, nstokes]

    The input spectra doesn't have to start from ell=0,
    and the output matrix spectra will start from the same lmin as the input spectra

    Parameters
    ----------
    :param c_ells_input: array of shape (n_correlations, lmax)

    Returns
    -------
    :return: reduced_matrix: array of shape (lmax+1-lmin, nstokes, nstokes)
    """
    c_ells_array = jnp.copy(c_ells_input)
    n_correlations = c_ells_array.shape[0]
    lmax_p1 = c_ells_array.shape[1]

    # Getting number of Stokes parameters from the number of correlations within the input spectrum
    if n_correlations == 1:
        nstokes = 1
    elif n_correlations == 3:
        nstokes = 2
    elif n_correlations == 4 or n_correlations == 6:
        nstokes = 3
        if n_correlations != 6:
            # c_ells_array = jnp.vstack(
            #     (c_ells_array, jnp.repeat(jnp.zeros(lmax_p1), 6 - n_correlations))
            # )
            c_ells_array = jnp.vstack(
                (c_ells_array, jnp.broadcast_to(jnp.zeros(lmax_p1), (6 - n_correlations, lmax_p1)).ravel(order='F'))
            )
            n_correlations = 6
    else:
        raise Exception(
            'C_ells must be given as TT for temperature only ; EE, BB, EB for polarization only ; TT, EE, BB, TE, (TB, EB) for both temperature and polarization'
        )

    # Constructing the reduced matrix
    reduced_matrix = jnp.zeros((lmax_p1, nstokes, nstokes))

    ## First diagonal elements
    def fmap(i, j):
        return jnp.einsum('l,sk->lsk', c_ells_array[i, :], jnp.eye(nstokes))[:, j]

    reduced_matrix = reduced_matrix.at[:, :, :].set(
        jax.vmap(fmap, in_axes=(0), out_axes=(1))(jnp.arange(nstokes), jnp.arange(nstokes))
    )

    ## Then off-diagonal elements
    if n_correlations > 1:
        reduced_matrix = reduced_matrix.at[:, 0, 1].set(c_ells_array[nstokes, :])
        reduced_matrix = reduced_matrix.at[:, 1, 0].set(c_ells_array[nstokes, :])
    if n_correlations == 6:
        reduced_matrix = reduced_matrix.at[:, 0, 2].set(c_ells_array[5, :])
        reduced_matrix = reduced_matrix.at[:, 2, 0].set(c_ells_array[5, :])

        reduced_matrix = reduced_matrix.at[:, 1, 2].set(c_ells_array[4, :])
        reduced_matrix = reduced_matrix.at[:, 2, 1].set(c_ells_array[4, :])

    return reduced_matrix


def get_c_ells_from_red_covariance_matrix_JAX(red_cov_mat, nstokes=0):
    """
    Retrieve the c_ell in the format [number_correlations, lmax+1-lmin],
    from the reduced covariance matrix format [lmax+1-lmin, nstokes, nstokes],
    assuming it's block diagonal

    Depending of nstokes, the number of correlations corresponds to:
        TT
        EE, BB, EB
        TT, EE, BB, TE, EB, TB
    ATTENTION : Currently not optimised for JAX (the for loops must be replaced by JAX loops)

    Parameters
    ----------
    :param red_cov_mat: reduced spectra of shape (lmax+1-lmin, nstokes, nstokes)

    Returns
    -------
    :return: c_ells: array of shape (n_correlations, lmax+1-lmin)
    """

    lmax = red_cov_mat.shape[0]
    nstokes = jnp.where(nstokes == 0, red_cov_mat.shape[1], nstokes)

    n_correl = jnp.int32(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))
    c_ells = jnp.zeros((n_correl, lmax))

    for i in range(nstokes):
        c_ells = c_ells.at[i, :].set(red_cov_mat[:, i, i])
    if nstokes > 1:
        c_ells = c_ells.at[nstokes, :].set(red_cov_mat[:, 0, 1])
        if nstokes == 3:
            c_ells = c_ells.at[nstokes + 2, :].set(red_cov_mat[:, 0, 2])
            c_ells = c_ells.at[nstokes + 1, :].set(red_cov_mat[:, 1, 2])
    return c_ells


def get_sqrt_reduced_matrix_from_matrix_jax(red_matrix):
    """
    Return matrix square root of covariance matrix in the format [lmax+1-lmin, nstokes, nstokes],
    assuming it's block diagonal

    The input matrix doesn't have to start from ell=0,
    and the output matrix will start from the same lmin as the input matrix

    The initial matrix HAVE to be positive semi-definite

    Parameters
    ----------
    :param red_matrix: reduced spectra of shape (lmax, nstokes, nstokes)

    Returns
    -------
    :return: reduced_sqrtm: array of shape (lmax, nstokes, nstokes)
    """

    red_matrix = jnp.array(red_matrix, dtype=jnp.float64)
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    reduced_sqrtm = jnp.zeros_like(red_matrix)

    # Building the square root matrix from the eigenvalues of the initial one
    eigvals, eigvect = jnp.linalg.eigh(red_matrix)
    inv_eigvect = jnp.linalg.pinv(eigvect)
    reduced_sqrtm = jnp.einsum('ljk,km,lm,lmn->ljn', eigvect, jnp.eye(nstokes), jnp.sqrt(jnp.abs(eigvals)), inv_eigvect)
    return reduced_sqrtm


def get_cell_from_map_jax(pixel_maps, lmax, n_iter=8):
    """
    Return c_ell from pixel_maps with an associated lmax and iteration number of harmonic operations

    Parameters
    ----------
    :param pixel_maps: array of shape (nstokes, n_pix)
    :param lmax: maximum ell for the spectrum, int
    :param n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    :return: c_ells: array of shape (nstokes, lmax+1)
    """

    # Wrapper for anafast, to prepare the pure callback of JAX
    def wrapper_anafast(maps_, lmax=lmax, n_iter=n_iter):
        return hp.anafast(maps_, lmax=lmax, iter=n_iter)

    # Pure call back of anafast, to be used with JAX for JIT compilation
    @partial(jax.jit, static_argnums=1)
    def pure_call_anafast(maps_, lmax):
        """Pure call back of anafast, to be used with JAX for JIT compilation"""
        shape_output = (6, lmax + 1)
        return jax.pure_callback(wrapper_anafast, jax.ShapeDtypeStruct(shape_output, np.float64), maps_)

    # Getting nstokes from the input maps
    if jnp.size(pixel_maps.shape) == 1:
        nstokes = 1
    else:
        nstokes = pixel_maps.shape[0]

    # Extending the pixel maps if they are given with only polarization Stokes parameters (nstokes=2)
    if nstokes == 2:
        pixel_maps_for_Wishart = jnp.vstack((jnp.zeros_like(pixel_maps[0]), pixel_maps))
    else:
        pixel_maps_for_Wishart = jnp.copy(pixel_maps)

    c_ells_output = pure_call_anafast(pixel_maps_for_Wishart, lmax=lmax)

    if nstokes == 2:
        polar_indexes = jnp.array([1, 2, 4])
        return c_ells_output[polar_indexes]  # Return only polarization spectra if nstokes=2
    return c_ells_output


@partial(jax.jit, static_argnames=('lmax'))
def alm_dot_product_JAX(alm_1, alm_2, lmax):
    """
    Return dot product of two alms

    Parameters
    ----------
    :param alm_1: input alms of shape (...,(lmax + 1) * (lmax // 2 + 1))
    :param alm_2: input alms of shape (...,(lmax + 1) * (lmax // 2 + 1))
    :param lmax: maximum ell for the spectrum, int

    Returns
    -------
    :return: dot_product: dot product of the two alms
    """

    real_part = alm_1.real * alm_2.real
    imag_part = alm_1.imag * alm_2.imag

    mask_true_m_contribution = jnp.where(jnp.arange(alm_1.shape[-1]) < lmax + 1, 1, 2)
    # See https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.Alm.getidx.html#healpy.sphtfunc.Alm.getidx
    # In HEALPix C++ and healpy, coefficients are stored ordered by m
    # So the first [lmax+1] elements of the alm array are the m=0 coefficients,
    # the next [lmax] are the m=1 coefficients, the following [lmax-1] are the m=2 coefficients,
    # the next [lmax-2] are the m=3 coefficients, etc.
    # and so on until the last element of the array which is the m=lmax coefficient.
    return jnp.sum((real_part + imag_part) * mask_true_m_contribution)


@partial(jax.jit, static_argnames=('lmax'))
def JAX_almxfl(alm, c_ell_x_, lmax):
    """
    Return alms convolved with the covariance matrix c_ell_x_ given as input in the format [lmax+1-lmin, nstokes, nstokes],
    assuming it's block diagonal, without the need of a pure callback to Healpy

    Parameters
    ----------
    :param alm: input alms of shape ((lmax + 1) * (lmax // 2 + 1))
    :param c_ell_x_: input spectra of shape (lmax+1)

    Returns
    -------
    :return: alms_output: output alms of shape ((lmax + 1) * (lmax // 2 + 1))
    """

    # Identifying the m indices of a set of alms according to Healpy convention
    all_m_idx = jax.vmap(lambda m_idx: m_idx * (2 * lmax + 1 - m_idx) // 2)(jnp.arange(lmax + 1))

    def func_scan(carry, ell):
        """
        For a given ell, returns the alms convolved with the covariance matrix c_ell_x_ for all m
        """
        _alm_carry = carry
        mask_m = jnp.where(jnp.arange(lmax + 1) <= ell, c_ell_x_[ell], 1)
        _alm_carry = _alm_carry.at[all_m_idx + ell].set(_alm_carry[all_m_idx + ell] * mask_m)
        return _alm_carry, ell

    alms_output, _ = jax.lax.scan(func_scan, jnp.copy(alm), jnp.arange(lmax + 1))
    return alms_output


def maps_x_red_covariance_cell_JAX(maps_input, red_matrix_sqrt, nside, lmin, n_iter=8):
    """
    Return maps convolved with the harmonic covariance matrix given as input
    in the format [lmax+1-lmin, nstokes, nstokes], assuming it's block diagonal

    The input matrix have to start from ell=lmin, otherwise the lmax associated with the harmonic
    operations will be wrong

    Parameters
    ----------
    :param maps_input: input maps of shape (nstokes, n_pix)
    :param red_matrix_sqrt: input reduced spectra of shape (lmax+1-lmin, nstokes, nstokes)
    :param nside: nside of the input maps, int
    :param lmin: minimum ell for the spectrum, int
    :param n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    :return: maps_output: input maps convolved with input spectra, dimensions (nstokes, n_pix)
    """

    # Getting scalar parameters from the input covariance
    all_params = 3
    lmax = red_matrix_sqrt.shape[0] - 1 + lmin
    nstokes = red_matrix_sqrt.shape[1]

    # Building the full covariance matrix from the covariance matrix
    red_decomp = jnp.zeros((lmax + 1, 3, 3))  # 3 is the maximum number of stokes parameters
    if nstokes != 1:
        red_decomp = red_decomp.at[lmin:, 3 - nstokes :, 3 - nstokes :].set(red_matrix_sqrt)
    else:
        red_decomp = red_decomp.at[lmin:].set(red_matrix_sqrt)

    # Extending the pixel maps if they are given with only polarization Stokes parameters (nstokes=2)
    if maps_input.shape[0] == 2:
        maps_TQU = jnp.vstack((jnp.zeros_like(maps_input[0]), jnp.copy(maps_input)))
    else:
        maps_TQU = jnp.copy(maps_input)

    # Wrapper for map2alm, to prepare the pure callback of JAX
    def wrapper_map2alm(maps_, lmax=lmax, n_iter=n_iter, nside=nside):
        alm_T, alm_E, alm_B = hp.map2alm(maps_.reshape((3, 12 * nside**2)), lmax=lmax, iter=n_iter)
        return np.array([alm_T, alm_E, alm_B])

    # Wrapper for alm2map, to prepare the pure callback of JAX
    def wrapper_alm2map(alm_, lmax=lmax, nside=nside):
        return hp.alm2map(alm_, nside, lmax=lmax)

    # Pure call back of map2alm, to be used with JAX for JIT compilation
    @partial(jax.jit, static_argnums=(1, 2))
    def pure_call_map2alm(maps_, lmax, nside):
        shape_output = (3, (lmax + 1) * (lmax // 2 + 1))
        return jax.pure_callback(
            wrapper_map2alm,
            jax.ShapeDtypeStruct(shape_output, np.complex128),
            maps_.ravel(),
        )

    @partial(jax.jit, static_argnums=(1, 2))
    def pure_call_alm2map(alm_, lmax, nside):
        shape_output = (3, 12 * nside**2)
        return jax.pure_callback(wrapper_alm2map, jax.ShapeDtypeStruct(shape_output, np.float64), alm_)

    alms_input = pure_call_map2alm(maps_TQU, lmax=lmax, nside=nside)

    # Multiplying the nstokes's jth alms with the covariance matrix for each stokes parameter contribution
    def scan_func(carry, nstokes_j):
        val_alms_j, nstokes_i = carry

        result_callback = JAX_almxfl(alms_input[nstokes_j], red_decomp[:, nstokes_i, nstokes_j], lmax)
        new_carry = (val_alms_j + result_callback, nstokes_i)
        return new_carry, val_alms_j + result_callback

    # Multiplying the nstokes's ith alms with the covariance matrix
    def fmap(nstokes_i):
        return jlax.scan(
            scan_func,
            (jnp.zeros_like(alms_input[nstokes_i]), nstokes_i),
            jnp.arange(all_params),
        )[
            0
        ][0]

    # Multiplying the alms with the covariance matrix
    alms_output = jax.vmap(fmap, in_axes=0)(jnp.arange(all_params))

    # Retrieving the maps from the alms convolved with the input covariance matrix
    maps_output = pure_call_alm2map(alms_output, nside=nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3 - nstokes :, ...]  # If only polarization maps are given, return only polarization maps
    return maps_output


def alms_x_red_covariance_cell_JAX(alm_Stokes_input, red_matrix, lmin):
    """
    Return alms convolved with the input harmonic covariance matrix in the format [lmax+1-lmin, nstokes, nstokes]
    given as input, assuming it's block diagonal

    The input matrix have to start from ell=lmin, otherwise the lmax associated with the harmonic
    operations will be wrong

    Parameters
    ----------
    :param alms_Stokes_input: input alms of shape (nstokes, (lmax + 1) * (lmax // 2 + 1)))
    :param red_matrix: input reduced spectra of shape (lmax+1-lmin, nstokes, nstokes)
    :param lmin: minimum ell for the spectrum, int

    Returns
    -------
    :return: maps_output: output maps of shape (nstokes, n_pix)
    """

    # Getting scalar parameters from the input covariance
    lmax = red_matrix.shape[0] - 1 + lmin
    nstokes = red_matrix.shape[1]

    # Building the full covariance matrix from the covariance matrix
    red_decomp = jnp.zeros((lmax + 1, nstokes, nstokes))

    # Preparing the alms and the covariance matrix for the convolution
    alm_input = jnp.copy(alm_Stokes_input)
    red_decomp = red_decomp.at[lmin:].set(red_matrix)

    # Multiplying the alms with the covariance matrix for each stokes parameter contribution
    def scan_func(carry, nstokes_j):
        """
        For a given nstokes_j, returns the alms convolved with the covariance matrix to be summed up for all nstokes_i
        """
        val_alms_j, nstokes_i = carry

        result_callback = JAX_almxfl(alm_input[nstokes_j], red_decomp[:, nstokes_i, nstokes_j], lmax)

        new_carry = (val_alms_j + result_callback, nstokes_i)
        return new_carry, val_alms_j + result_callback

    # Multiplying the ie alms with the covariance matrix
    def fmap(nstokes_i):
        return jlax.scan(
            scan_func,
            (jnp.zeros_like(alm_input[0]), nstokes_i),
            jnp.arange(nstokes),
        )[
            0
        ][0]

    # Multiplying the alms with the covariance matrix
    alms_output = jax.vmap(fmap, in_axes=0)(jnp.arange(nstokes))

    return alms_output


def frequency_alms_x_obj_red_covariance_cell_JAX(freq_alm_Stokes_input, freq_red_matrix, lmin, n_iter=8):
    """
    Return frequency alms convolved with the covariance matrix given as input, assuming it's block diagonal

    The freq_red_matrix can have its first dimension reprensenting anything,
    in particular either the number of frequencies or the number of components

    Parameters
    ----------
    :param freq_alm_Stokes_input: input alms of shape [frequency, nstokes, (lmax + 1) * (lmax // 2 + 1)))
    :param red_matrix: input reduced spectra of shape [first_dim, frequency, lmax+1-lmin, nstokes, nstokes]
    :param lmin: minimum ell for the spectrum, int
    :param n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    :return: maps_output: output maps of shape (nstokes, n_pix)
    """

    # Getting scalar parameters from the input covariance
    lmax = freq_red_matrix.shape[2] - 1 + lmin
    first_dim_red_matrix = freq_red_matrix.shape[
        0
    ]  # Can be any dimension, in the use of the function either n_frequencies or n_components
    n_frequencies = freq_red_matrix.shape[1]
    nstokes = freq_red_matrix.shape[3]

    # Few tests to check the input
    chx.assert_axis_dimension(freq_red_matrix, 1, n_frequencies)
    chx.assert_axis_dimension(freq_red_matrix, 2, lmax + 1 - lmin)
    chx.assert_axis_dimension(freq_red_matrix, 3, nstokes)
    chx.assert_axis_dimension(freq_red_matrix, 4, nstokes)
    chx.assert_shape(freq_alm_Stokes_input, (n_frequencies, nstokes, (lmax + 1) * (lmax // 2 + 1)))

    freq_alm_input = jnp.copy(freq_alm_Stokes_input)

    def scan_func(carry, frequency_j):
        """
        For a given frequency_j, returns the alms convolved with the frequency covariance matrix to be summed up for all nstokes_i
        """
        val_alms_j, idx_i = carry
        result_callback = alms_x_red_covariance_cell_JAX(
            freq_alm_input[frequency_j], freq_red_matrix[idx_i, frequency_j, ...], lmin=lmin
        )
        new_carry = (val_alms_j + result_callback, idx_i)
        return new_carry, val_alms_j + result_callback

    # Multiplying the ie alms with the covariance matrix
    def fmap(idx_i):
        """
        For a given idx_i, returns the alms convolved with the frequency covariance matrix to be summed up for all corresponding frequencies
        """
        return jlax.scan(
            scan_func,
            (jnp.zeros_like(freq_alm_input[0]), idx_i),
            jnp.arange(n_frequencies),
        )[
            0
        ][0]

    # Multiplying the frequency alms with the first dimension-frequency covariance matrix
    freq_alms_output = jax.vmap(fmap, in_axes=0)(jnp.arange(first_dim_red_matrix))
    return freq_alms_output
