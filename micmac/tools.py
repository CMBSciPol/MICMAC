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
    :param c_ells_input: array of shape (n_correlations, lmax)

    Returns
    -------
    :return: reduced_matrix: array of shape (lmax+1-lmin, nstokes, nstokes)
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
    :param red_matrix: reduced spectra of shape (lmax, nstokes, nstokes)

    Returns
    -------
    :return: reduced_sqrtm: array of shape (lmax, nstokes, nstokes)
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
    :param pixel_maps: array of shape (nstokes, n_pix)
    :param lmax: maximum ell for the spectrum, int
    :param n_iter: number of iterations for harmonic operations, int

    Returns
    -------
    :return: c_ells: array of shape (nstokes, lmax+1)
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
