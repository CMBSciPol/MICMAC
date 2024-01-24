import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jlax
import healpy as hp
from functools import partial


def get_reduced_matrix_from_c_ell_jax(c_ells_input):
    """Expect c_ells_input to be sorted as TT, EE, BB, TE, TB, EB if 6 spectra are given
        or EE, BB, EB if 3 spectra are given
        or TT if 1 spectrum is given

    Generate covariance matrix from c_ells assuming it's block diagonal,
    in the "reduced" (prefix red) format, i.e. : [ell, nstokes, nstokes]

    The input spectra doesn't have to start from ell=0,
    and the output matrix spectra will start from the same lmin as the input spectra

    Parameters
    ----------
    :param c_ells_input: array of shape (number_correlations, lmax)

    Returns
    -------
    :return: reduced_matrix: array of shape (lmax, nstokes, nstokes)
    """
    c_ells_array = jnp.copy(c_ells_input)
    number_correlations = c_ells_array.shape[0]
    # assert number_correlations == 1 or number_correlations == 3 or number_correlations == 6
    lmax_p1 = c_ells_array.shape[1]

    # Getting number of Stokes parameters from the number of correlations within the input spectrum
    if number_correlations == 1:
        nstokes = 1
    elif number_correlations == 3:
        nstokes = 2
    elif number_correlations == 4 or number_correlations == 6:
        nstokes = 3
        if number_correlations != 6:
            c_ells_array = jnp.vstack(
                (c_ells_array, jnp.repeat(jnp.zeros(lmax_p1), 6 - number_correlations))
            )
            number_correlations = 6
    else:
        raise Exception(
            "C_ells must be given as TT for temperature only ; EE, BB, EB for polarization only ; TT, EE, BB, TE, (TB, EB) for both temperature and polarization"
        )

    # Constructing the reduced matrix
    reduced_matrix = jnp.zeros((lmax_p1, nstokes, nstokes))

    ## First diagonal elements
    def fmap(i, j):
        return jnp.einsum("l,sk->lsk", c_ells_array[i, :], jnp.eye(nstokes))[:, j]

    reduced_matrix = reduced_matrix.at[:, :, :].set(
        jax.vmap(fmap, in_axes=(0), out_axes=(1))(
            jnp.arange(nstokes), jnp.arange(nstokes)
        )
    )

    # for i in range(nstokes):
    #     # reduced_matrix[:,i,i] =  c_ells_array[i,:]
    #     reduced_matrix = reduced_matrix.at[:,i,i].set(c_ells_array[i,:])

    ## Then off-diagonal elements
    if number_correlations > 1:
        reduced_matrix = reduced_matrix.at[:, 0, 1].set(c_ells_array[nstokes, :])
        reduced_matrix = reduced_matrix.at[:, 1, 0].set(c_ells_array[nstokes, :])
    if number_correlations == 6:
        reduced_matrix = reduced_matrix.at[:, 0, 2].set(c_ells_array[5, :])
        reduced_matrix = reduced_matrix.at[:, 2, 0].set(c_ells_array[5, :])

        reduced_matrix = reduced_matrix.at[:, 1, 2].set(c_ells_array[4, :])
        reduced_matrix = reduced_matrix.at[:, 2, 1].set(c_ells_array[4, :])

    return reduced_matrix


def get_c_ells_from_red_covariance_matrix_JAX(red_cov_mat, nstokes=0):
    """Retrieve c_ell from red_cov_mat, which depending of nstokes will give :
    TT
    EE, BB, EB
    TT, EE, BB, TE, EB, TB

    """

    lmax = red_cov_mat.shape[0]
    nstokes = jnp.where(nstokes == 0, red_cov_mat.shape[1], nstokes)

    number_correl = jnp.int32(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))
    # number_correl = jnp.array(jnp.ceil(nstokes**2/2) + jnp.floor(nstokes/2),int)
    c_ells = jnp.zeros((number_correl, lmax))

    for i in range(nstokes):
        c_ells = c_ells.at[i, :].set(red_cov_mat[:, i, i])
    if nstokes > 1:
        c_ells = c_ells.at[nstokes, :].set(red_cov_mat[:, 0, 1])
        if nstokes == 3:
            c_ells = c_ells.at[nstokes + 2, :].set(red_cov_mat[:, 0, 2])
            c_ells = c_ells.at[nstokes + 1, :].set(red_cov_mat[:, 1, 2])
    return c_ells


# def get_sqrt_reduced_matrix_from_matrix_jax(red_matrix):
#     """Return square root matrix of red_matrix, assuming it's block diagonal

#     The input matrix doesn't have to start from ell=0,
#     and the output matrix will start from the same lmin as the input matrix

#     The initial matrix HAVE to be positive semi-definite

#     Parameters
#     ----------
#     :param red_matrix: reduced spectra of shape (lmax, nstokes, nstokes)

#     Returns
#     -------
#     :return: reduced_sqrtm: array of shape (lmax, nstokes, nstokes)
#     """

#     red_matrix = jnp.array(red_matrix, dtype=jnp.float64)
#     lmax = red_matrix.shape[0]
#     nstokes = red_matrix.shape[1]

#     reduced_sqrtm = jnp.zeros_like(red_matrix)

#     # Building the square root matrix from the eigenvalues of the initial one
#     def func_map(ell):
#         eigvals, eigvect = jnp.linalg.eigh(red_matrix[ell, :, :])
#         inv_eigvect = jnp.array(jnp.linalg.pinv(eigvect), dtype=jnp.float64)
#         return jnp.einsum(
#             "jk,km,m,mn->jn",
#             eigvect,
#             jnp.eye(nstokes),
#             jnp.sqrt(jnp.abs(eigvals)),
#             inv_eigvect,
#         )

#     reduced_sqrtm = jax.vmap(func_map, in_axes=0)(jnp.arange(lmax))
#     return reduced_sqrtm

def get_sqrt_reduced_matrix_from_matrix_jax(red_matrix):
    """Return square root matrix of red_matrix, assuming it's block diagonal

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
    reduced_sqrtm = jnp.einsum(
        "ljk,km,lm,lmn->ljn",
        eigvect,
        jnp.eye(nstokes),
        jnp.sqrt(jnp.abs(eigvals)),
        inv_eigvect,
        )

    # reduced_sqrtm = jax.vmap(func_map, in_axes=0)(jnp.arange(lmax))
    return reduced_sqrtm

def get_cell_from_map_jax(pixel_maps, lmax, n_iter=8):
    """Return c_ell from pixel_maps with an associated lmax and iteration number of harmonic operations

    Parameters
    ----------
    :param pixel_maps: array of shape (nstokes, npix)
    :param lmax: maximum ell for the spectrum, int
    :param n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    :return: c_ells: array of shape (nstokes, lmax+1)
    """

    # Wrapper for anafast, to prepare the pure callback of JAX
    def wrapper_anafast(maps_, lmax=lmax, n_iter=n_iter):
        return hp.anafast(maps_, lmax=lmax, iter=n_iter)

    @partial(jax.jit, static_argnums=1)
    def pure_call_anafast(maps_, lmax):
        """Pure call back of anafast, to be used with JAX for JIT compilation"""
        shape_output = (6, lmax + 1)
        return jax.pure_callback(
            wrapper_anafast, jax.ShapeDtypeStruct(shape_output, np.float64), maps_
        )

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

    # c_ells_output = hp.anafast(pixel_maps_for_Wishart, lmax=lmax, iter=n_iter)
    c_ells_output = pure_call_anafast(pixel_maps_for_Wishart, lmax=lmax)

    if nstokes == 2:
        polar_indexes = jnp.array([1, 2, 4])
        return c_ells_output[
            polar_indexes
        ]  # Return only polarization spectra if nstokes=2
    return c_ells_output


def maps_x_red_covariance_cell_JAX(
    maps_TQU_input, red_matrix_sqrt, nside, lmin, n_iter=8
):
    """Return maps convolved with the covariance matrix given as input, assuming it's block diagonal

    The input matrix have to start from ell=lmin, otherwise the lmax associated with the harmonic
    operations will be wrong

    Parameters
    ----------
    :param maps_TQU_input: input maps of shape (nstokes, npix)
    :param red_matrix_sqrt: input reduced spectra of shape (lmax+1-lmin, nstokes, nstokes)
    :param nside: nside of the input maps, int
    :param lmin: minimum ell for the spectrum, int
    :param n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    :return: maps_output: output maps of shape (nstokes, npix)
    """

    all_params = 3

    # Getting scalar parameters from the input covariance
    lmax = red_matrix_sqrt.shape[0] - 1 + lmin
    nstokes = red_matrix_sqrt.shape[1]

    # Building the full covariance matrix from the covariance matrix
    red_decomp = jnp.zeros((lmax + 1, all_params, all_params))
    if nstokes != 1:
        red_decomp = red_decomp.at[lmin:, 3 - nstokes :, 3 - nstokes :].set(
            red_matrix_sqrt
        )
    else:
        red_decomp = red_decomp.at[lmin:].set(red_matrix_sqrt)

    # Extending the pixel maps if they are given with only polarization Stokes parameters (nstokes=2)
    if maps_TQU_input.shape[0] == 2:
        maps_TQU = jnp.vstack(
            (jnp.zeros_like(maps_TQU_input[0]), jnp.copy(maps_TQU_input))
        )
    else:
        maps_TQU = jnp.copy(maps_TQU_input)

    # Wrapper for map2alm, to prepare the pure callback of JAX
    def wrapper_map2alm(maps_, lmax=lmax, n_iter=n_iter, nside=nside):
        alm_T, alm_E, alm_B = hp.map2alm(
            maps_.reshape((3, 12 * nside**2)), lmax=lmax, iter=n_iter
        )
        return np.array([alm_T, alm_E, alm_B])

    # Wrapper for almxfl, to prepare the pure callback of JAX
    def wrapper_almxfl(alm_, matrix_ell):
        return hp.almxfl(alm_, matrix_ell, inplace=False)

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

    def pure_call_almxfl(alm_, matrix_ell):
        shape_output = [(lmax + 1) * (lmax // 2 + 1)]
        return jax.pure_callback(
            wrapper_almxfl,
            jax.ShapeDtypeStruct(shape_output, np.complex128),
            alm_,
            matrix_ell,
        )

    @partial(jax.jit, static_argnums=(1, 2))
    def pure_call_alm2map(alm_, lmax, nside):
        shape_output = (3, 12 * nside**2)
        return jax.pure_callback(
            wrapper_alm2map, jax.ShapeDtypeStruct(shape_output, np.float64), alm_
        )

    alms_input = pure_call_map2alm(maps_TQU, lmax=lmax, nside=nside)

    # Multiplying the je alms with the covariance matrix for each stokes parameter contribution
    def scan_func(carry, nstokes_j):
        val_alms_j, nstokes_i = carry
        result_callback = pure_call_almxfl(
            alms_input[nstokes_j], red_decomp[:, nstokes_i, nstokes_j]
        )
        new_carry = (val_alms_j + result_callback, nstokes_i)
        return new_carry, val_alms_j + result_callback

    # Multiplying the ie alms with the covariance matrix
    def fmap(nstokes_i):
        return jlax.scan(
            scan_func,
            (jnp.zeros_like(alms_input[nstokes_i]), nstokes_i),
            jnp.arange(all_params),
        )[0][0]

    # Multiplying the alms with the covariance matrix
    alms_output = jax.vmap(fmap, in_axes=0)(jnp.arange(all_params))

    maps_output = pure_call_alm2map(alms_output, nside=nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[
            3 - nstokes :, ...
        ]  # If only polarization maps are given, return only polarization maps
    return maps_output
