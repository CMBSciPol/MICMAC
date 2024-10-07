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

__all__ = [
    'get_reduced_matrix_from_c_ell_jax',
    'get_c_ells_from_red_covariance_matrix',
    'get_c_ells_from_red_covariance_matrix_JAX',
    'get_sqrt_reduced_matrix_from_matrix_jax',
    'get_cell_from_map_jax',
    'get_bool_array_in_boundary',
    'alm_dot_product_JAX',
    'JAX_almxfl',
    'maps_x_red_covariance_cell_JAX',
    'alms_x_red_covariance_cell_JAX',
    'frequency_alms_x_obj_red_covariance_cell_JAX',
]


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
    c_ells_input: array of shape (n_correlations, lmax)
        Input c_ells

    Returns
    -------
    reduced_matrix: array of shape (lmax+1-lmin, nstokes, nstokes)
        Reduced format of the covariance matrix
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
    red_cov_mat: array[float] of dimensions [lmax+1-lmin, nstokes, nstokes]
        reduced spectra of the covariance matrix

    Returns
    -------
    c_ells: array of dimensions [n_correlations, lmax+1-lmin]
        power specturm of the input reduced covariance matrix
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
    red_matrix: array of dimensions [lmax+1-lmin, nstokes, nstokes]
        reduced spectra of the covariance matrix

    Returns
    -------
    reduced_sqrtm: array of dimensions [lmax+1-lmin, nstokes, nstokes]
        matrix square root of the covariance matrix
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
    pixel_maps: array of dimensions [nstokes, n_pix]
        input maps
    lmax: int
        maximum ell for the spectrum
    n_iter: int
        number of iterations for harmonic operations

    Returns
    -------
    c_ells: array of dimensions[n_correlations,lmin:lmax+1]
        power specturm of the input maps
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


def get_bool_array_in_boundary(input_array, boundary):
    """
    Return a boolean array of the same shape as the input array, with True values where the input array is within the boundary

    Parameters
    ----------
    input_array: array
        array to test
    boundary: array of dimension [2,dim(input_array)]
        represents the boundary

    Returns
    -------
    bool_array: array[bool]
        boolean array of the same shape as the input array,
        with True values where the input array is within the boundary
    """
    return (input_array >= boundary[0]) & (input_array <= boundary[1])


@partial(jax.jit, static_argnames=('lmax'))
def alm_dot_product_JAX(alm_1, alm_2, lmax):
    """
    Return dot product of two alms

    Parameters
    ----------
    alm_1: array
        input alms of shape (...,(lmax + 1) * (lmax // 2 + 1))
    alm_2: array
        input alms of shape (...,(lmax + 1) * (lmax // 2 + 1))
    lmax: int
        maximum ell for the power spectrum

    Returns
    -------
    dot_product: float
        dot product of the two alms
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
    alm:  array
        input alms of shape ((lmax + 1) * (lmax // 2 + 1))
    c_ell_x_: array of shape [lmax+1]
        input power spectrum

    Returns
    -------
    alms_output: array
        updated output alms of shape ((lmax + 1) * (lmax // 2 + 1))
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
    maps_input: array[float] of shape [nstokes, n_pix]
         input maps
    red_matrix_sqrt: array[float] of shape [lmax+1-lmin, nstokes, nstokes]
        input reduced spectra
    nside: int
        nside of the input maps
    lmin: int
        minimum ell for the spectrum
    n_iter: int
        number of iterations for harmonic operations

    Returns
    -------
    maps_output: array[float] of shape [nstokes, n_pix]
        input maps convolved with input spectra
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
    alms_Stokes_input: arary of shape [nstokes, (lmax + 1) * (lmax // 2 + 1)]
        input alms
    red_matrix: array of shape [lmax+1-lmin, nstokes, nstokes]
        input reduced covariance matrix
    lmin: int
        minimum ell for the spectrum

    Returns
    -------
    maps_output: array of shape [nstokes, n_pix]
        output maps
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
    freq_alm_Stokes_input: array of shape [frequency, nstokes, (lmax + 1) * (lmax // 2 + 1))]
         input alms per frequency
    red_matrix: array of shape [first_dim, frequency, lmax+1-lmin, nstokes, nstokes]
        input reduced covariance matrix
    lmin: int
        minimum ell for the power spectrum
    n_iter: int
        number of iterations for harmonic operations

    Returns
    -------
    maps_output: array[float] of shape [nstokes, n_pix]
        output maps
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


## Numpy version


import healpy as hp
import numpy as np


def get_reduced_matrix_from_c_ell(c_ells_input):
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
    c_ells_input: array of shape (n_correlations, lmax)
        input power spectra

    Returns
    -------
    reduced_matrix: array of shape (lmax+1-lmin, nstokes, nstokes)
        reduced covariance matrix
    """
    c_ells_array = np.copy(c_ells_input)
    n_correlations = c_ells_array.shape[0]
    assert n_correlations == 1 or n_correlations == 3 or n_correlations == 6
    lmax_p1 = c_ells_array.shape[1]
    if n_correlations == 1:
        nstokes = 1
    elif n_correlations == 3:
        nstokes = 2

    elif n_correlations == 4 or n_correlations == 6:
        nstokes = 3
        if n_correlations != 6:
            for i in range(6 - n_correlations):
                c_ells_array = np.vstack((c_ells_array, np.zeros(lmax_p1)))
            n_correlations = 6
    else:
        raise Exception(
            'C_ells must be given as TT for temperature only ; EE, BB, EB for polarization only ; TT, EE, BB, TE, (TB, EB) for both temperature and polarization'
        )

    reduced_matrix = np.zeros((lmax_p1, nstokes, nstokes))

    for i in range(nstokes):
        reduced_matrix[:, i, i] = c_ells_array[i, :]

    # for j in range(n_correlations-nstokes):
    if n_correlations > 1:
        reduced_matrix[:, 0, 1] = c_ells_array[nstokes, :]
        reduced_matrix[:, 1, 0] = c_ells_array[nstokes, :]

    if n_correlations == 6:
        reduced_matrix[:, 0, 2] = c_ells_array[5, :]
        reduced_matrix[:, 2, 0] = c_ells_array[5, :]

        reduced_matrix[:, 1, 2] = c_ells_array[4, :]
        reduced_matrix[:, 2, 1] = c_ells_array[4, :]

    return reduced_matrix


def get_c_ells_from_red_covariance_matrix(red_cov_mat):
    """
    Retrieve the c_ell in the format [number_correlations, lmax+1-lmin],
    from the reduced covariance matrix format [lmax+1-lmin, nstokes, nstokes],
    assuming it's block diagonal

    Depending of nstokes, the number of correlations corresponds to:
        TT
        EE, BB, EB
        TT, EE, BB, TE, EB, TB
    """

    lmax = red_cov_mat.shape[0]
    nstokes = red_cov_mat.shape[1]

    n_correl = int(np.ceil(nstokes**2 / 2) + np.floor(nstokes / 2))
    c_ells = np.zeros((n_correl, lmax))

    for i in range(nstokes):
        c_ells[i, :] = red_cov_mat[:, i, i]
    if nstokes > 1:
        c_ells[nstokes, :] = red_cov_mat[:, 0, 1]
        if nstokes == 3:
            # c_ells[nstokes+1,:] = red_cov_mat[:,0,2]
            # c_ells[nstokes+2,:] = red_cov_mat[:,1,2]
            c_ells[nstokes + 2, :] = red_cov_mat[:, 0, 2]
            c_ells[nstokes + 1, :] = red_cov_mat[:, 1, 2]
    return c_ells


def get_sqrt_reduced_matrix_from_matrix(red_matrix, tolerance=10 ** (-15)):
    """
    Return matrix square root of covariance matrix in the format [lmax+1-lmin, nstokes, nstokes],
    assuming it's block diagonal

    The input matrix doesn't have to start from ell=0,
    and the output matrix will start from the same lmin as the input matrix

    The initial matrix HAVE to be positive semi-definite

    Parameters
    ----------
    red_matrix: array of shape (lmax, nstokes, nstokes)
        reduced covariance matrix

    Returns
    -------
    reduced_sqrtm: array of shape (lmax, nstokes, nstokes)
        reduced matrix square root of the covariance matrix
    """

    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    reduced_sqrtm = np.zeros_like(red_matrix)

    for ell in range(red_matrix.shape[0]):
        eigvals, eigvect = np.linalg.eigh(red_matrix[ell, :, :])

        try:
            inv_eigvect = np.linalg.pinv(eigvect)
        except:
            raise Exception(
                'Error for ell=', ell, 'eigvals', eigvals, 'eigvect', eigvect, 'red_matrix', red_matrix[ell, :, :]
            )

        if not (np.all(eigvals > 0)) and (np.abs(eigvals[eigvals < 0]) > tolerance):
            raise Exception(
                'Covariance matrix not consistent with a negative eigval for ell=',
                ell,
                'eigvals',
                eigvals,
                'eigvect',
                eigvect,
                'red_matrix',
                red_matrix[ell, :, :],
            )

        reduced_sqrtm[ell] = np.einsum(
            'jk,km,m,mn->jn', eigvect, np.eye(nstokes), np.sqrt(np.abs(eigvals)), inv_eigvect
        )
    return reduced_sqrtm


def get_cell_from_map(pixel_maps, lmax, n_iter=8):
    """
    Return c_ell from pixel_maps with an associated lmax and iteration number of harmonic operations

    Parameters
    ----------
    pixel_maps: array of shape (nstokes, n_pix)
        input maps
    lmax: int
        maximum ell for the spectrum
    n_iter: int
        number of iterations for harmonic operations

    Returns
    -------
    c_ells: array of shape (nstokes, lmax+1)
        power spectra from the input maps
    """

    if len(pixel_maps.shape) == 1:
        nstokes = 1
    else:
        nstokes = pixel_maps.shape[0]

    if nstokes == 2:
        pixel_maps_for_Wishart = np.vstack((np.zeros_like(pixel_maps[0]), pixel_maps))
    else:
        pixel_maps_for_Wishart = pixel_maps

    c_ells_Wishart = hp.anafast(pixel_maps_for_Wishart, lmax=lmax, iter=n_iter)

    if nstokes == 2:
        polar_indexes = np.array([1, 2, 4])
        c_ells_Wishart = c_ells_Wishart[polar_indexes]
    return c_ells_Wishart


def maps_x_reduced_matrix_generalized_sqrt_sqrt(maps_TQU_input, red_matrix_sqrt, lmin, n_iter=8):
    """
    NOT USED -- TO BE REMOVED

    Return maps convolved with the harmonic covariance matrix given as input
    in the format [lmax+1-lmin, nstokes, nstokes], assuming it's block diagonal

    The input matrix have to start from ell=lmin, otherwise the lmax associated with the harmonic
    operations will be wrong

    Parameters
    ----------
    maps_input: input maps of shape (nstokes, n_pix)
    red_matrix_sqrt: input reduced spectra of shape (lmax+1-lmin, nstokes, nstokes)
    nside: nside of the input maps, int
    lmin: minimum ell for the spectrum, int
    n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    maps_output: input maps convolved with input spectra, dimensions (nstokes, n_pix)
    """
    lmax = red_matrix_sqrt.shape[0] - 1 + lmin
    nstokes = red_matrix_sqrt.shape[1]
    all_params = int(np.where(nstokes > 1, 3, 1))

    if len(maps_TQU_input.shape) == 1:
        nside = int(np.sqrt(len(maps_TQU_input) / 12))
    else:
        nside = int(np.sqrt(len(maps_TQU_input[0]) / 12))

    red_sqrt_decomp = np.zeros((lmax + 1, all_params, all_params))
    if nstokes != 1:
        red_sqrt_decomp[lmin:, 3 - nstokes :, 3 - nstokes :] = red_matrix_sqrt
    else:
        red_sqrt_decomp[lmin:, ...] = red_matrix_sqrt

    if maps_TQU_input.shape[0] == 2:
        maps_TQU = np.vstack((np.zeros_like(maps_TQU_input[0]), np.copy(maps_TQU_input)))
    else:
        maps_TQU = np.copy(maps_TQU_input)

    alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)

    alms_output = np.zeros_like(alms_input)

    for i in range(all_params):
        alms_j = np.zeros_like(alms_input[i])
        for j in range(all_params):
            alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:, i, j], inplace=False)
        alms_output[i] = np.copy(alms_j)
    maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3 - nstokes :, ...]
    return maps_output
