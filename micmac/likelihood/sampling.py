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

import time

import chex as chx
import jax
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import jax_healpy as jhp
import numpy as np
import numpyro.distributions as dist

from micmac.foregrounds.mixingmatrix import MixingMatrix
from micmac.noise.noisecovar import get_BtinvN, get_inv_BtinvNB, get_inv_BtinvNB_c_ell
from micmac.toolbox.tools import (
    alm_dot_product_JAX,
    alms_x_red_covariance_cell_JAX,
    frequency_alms_x_obj_red_covariance_cell_JAX,
    get_bool_array_in_boundary,
    get_reduced_matrix_from_c_ell_jax,
    get_sqrt_reduced_matrix_from_matrix_jax,
    maps_x_red_covariance_cell_JAX,
)

__all__ = [
    'Sampling_functions',
    'single_Metropolis_Hasting_step',
    'bounded_single_Metropolis_Hasting_step',
    'multivariate_Metropolis_Hasting_step',
    'multivariate_Metropolis_Hasting_step_numpyro',
    'multivariate_Metropolis_Hasting_step_numpyro_bounded',
    'separate_single_MH_step_index_v2b',
    'separate_single_MH_step_index_v3',
    'separate_single_MH_step_index_v4_pixel',
    'separate_single_MH_step_index_v4b_pixel',
    'separate_single_MH_step_index_accelerated',
]


class Sampling_functions(MixingMatrix):
    def __init__(
        self,
        nside,
        lmax,
        nstokes,
        frequency_array,
        freq_inverse_noise,
        pos_special_freqs=[0, -1],
        spv_nodes_b=None,
        freq_noise_c_ell=None,
        mask=None,
        n_components=3,
        lmin=2,
        n_iter=8,
        limit_iter_cg=200,
        limit_iter_cg_eta=200,
        tolerance_CG=1e-10,
        atol_CG=1e-8,
        bin_ell_distribution=None,
    ):
        """Sampling functions object
        Contains all the functions needed for the sampling step of the Gibbs sampler

        Parameters
        ----------
        nside: int
            nside of the input frequency maps
        lmax: int
            maximum multipole for the spherical harmonics transforms and harmonic domain objects,
        nstokes: int
            number of Stokes parameters
        frequency_array: array[float]
            array of frequencies, in GHz
        freq_inverse_noise: array[float]
            array of inverse noise for each frequency, in uK^-2
        pos_special_freqs: list[int] (optional)
            indexes of the special frequencies in the frequency array respectively for synchrotron and dust, default is [0,-1] for first and last frequencies
        spv_nodes_b: list[dictionaries] (optional)
            tree for the spatial variability, to generate from a yaml file, default None
            in principle set up by get_nodes_b
        freq_noise_c_ell: array[float] of dimensions [frequencies, frequencies, lmax+1-lmin] or [frequencies, frequencies, lmax] (in which case it will be cut to lmax+1-lmin) (optional)
            optional, noise power spectra for each frequency, in uK^2, dimensions, default None
        mask: None or array[float] of dimensions [n_pix] (optional)
            mask to use in the sampling  ; if not given, no mask is used, default None
            Note: the mask WILL NOT be applied to the input maps, it will be only used for the propagated noise covariance
        n_components: int (optional)
            number of components for the mixing matrix, default 3
        lmin: int (optional)
            minimum multipole for the spherical harmonics transforms and harmonic domain objects, default 2
        n_iter: int (optional)
            number of iterations the spherical harmonics transforms (for map2alm transformations), default 8
        limit_iter_cg: int (optional)
            maximum number of iterations for the conjugate gradient for the CMB map sampling, default 200
        limit_iter_cg_eta: int (optional)
            maximum number of iterations for the conjugate gradient for eta maps sampling, default 200
        tolerance_CG: float (optional)
            tolerance for the conjugate gradient, default 1e-8
        atol_CG: float (optional)
            absolute tolerance for the conjugate gradient, default 1e-8
        bin_ell_distribution: array[int] (optional)
            array of the bounds of the bins if binned Inverse Wishart considered, of size nbins+1, default None for no binning
        """

        # Inheritance from MixingMatrix
        super().__init__(
            nside=nside,
            frequency_array=frequency_array,
            n_components=n_components,
            params=None,
            pos_special_freqs=pos_special_freqs,
            spv_nodes_b=spv_nodes_b,
        )

        # Problem parameters
        assert nstokes == 2  # Focus on polarisation for now
        self.nstokes = int(nstokes)
        self.n_correlations = int(np.ceil(self.nstokes**2 / 2) + np.floor(self.nstokes / 2))
        # Number of correlations for the power spectrum, so if nstokes==2, it should 3 corresponding to [EE, BB, EB]
        self.nside = int(nside)  # nside of the input maps
        self.lmax = int(lmax)  # maximum ell to be considered
        assert lmin >= 2
        self.lmin = int(lmin)  # minimum ell to be considered

        # Noise parameters
        if freq_inverse_noise is not None:
            assert freq_inverse_noise.shape == (self.n_frequencies, self.n_frequencies, self.n_pix)
        self.freq_inverse_noise = freq_inverse_noise
        if freq_noise_c_ell is not None:
            assert freq_noise_c_ell.shape == (self.n_frequencies, self.n_frequencies, self.lmax + 1 - self.lmin)
        self.freq_noise_c_ell = freq_noise_c_ell  # If noise expected to be white, c_ell of the noise

        # Mask parameters
        if mask is None:
            # If no mask is given, then the mask is set to 1
            self.mask = np.ones(12 * self.nside**2)
        else:
            self.mask = mask
        if bin_ell_distribution is None:
            # If no binning, then the bin_ell_distribution is set to the full range of ell
            self.bin_ell_distribution = jnp.arange(self.lmin, self.lmax + 2)
        else:
            self.bin_ell_distribution = bin_ell_distribution  # Expects array of the bounds of the bins, of size nbins+1
        self.maximum_number_dof = int(self.bin_ell_distribution[-1] ** 2 - self.bin_ell_distribution[-2] ** 2)
        # Maximum number of degrees of freedom for the binned Inverse Wishart distribution

        # CG and harmonic parameters
        self.n_iter = int(n_iter)  # Number of iterations for estimation of alms with Healpy
        self.limit_iter_cg = int(
            limit_iter_cg
        )  # Maximum number of iterations for the CGs used in the CMB maps sampling
        self.limit_iter_cg_eta = float(
            limit_iter_cg_eta
        )  # Maximum number of iterations for the CG used in the eta sampling
        self.tolerance_CG = float(tolerance_CG)  # Tolerance for the different CGs
        self.atol_CG = float(atol_CG)  # Absolute tolerance for the different CGs

    @property
    def n_pix(self):
        """Return number of pixels of 1 map, for a given Stokes parameter"""
        return 12 * self.nside**2

    @property
    def number_bins(self):
        """Return number of bins"""
        return jnp.size(self.bin_ell_distribution) - 1

    def number_dof(self, bin_index):
        """Return number of degrees of freedom for a given bin, for inverse Wishart distribution"""
        return (self.bin_ell_distribution[bin_index + 1]) ** 2 - self.bin_ell_distribution[bin_index] ** 2

    def project_bin_to_ell(self, binned_spectra):
        """
        Project binned spectra by repeating the bin value for each ell

        Parameters
        ----------
        binned_spectra: array[float] of dimensions [number_bins, nstokes, nstokes]
            binned power spectra

        Returns
        -------
        ell_projection: array[float] of dimensions [lmax+1, nstokes, nstokes]
            dProjected power spectra
        """
        # chx.assert_axis_dimension(binned_spectra, 0, self.number_bins)

        ell_distribution = jnp.arange(self.lmin, self.lmax + 1)

        def scan_binned_c_ells(ell_projection, bin_idx):
            """Compute the binned power spectrum for a given ell"""
            cond_bool = jnp.logical_and(
                self.bin_ell_distribution[bin_idx] <= ell_distribution,
                self.bin_ell_distribution[bin_idx + 1] > ell_distribution,
            )
            ell_projection += jnp.where(cond_bool, binned_spectra[bin_idx], 0)
            return ell_projection, None

        ell_projection = jlax.scan(
            scan_binned_c_ells, jnp.zeros_like(ell_distribution, dtype=jnp.float64), jnp.arange(self.number_bins)
        )[0]
        # ell_projection = ell_projection.at[self.bin_ell_distribution[-1]:].set(binned_spectra[-1])
        return ell_projection

    def get_binned_c_ells(self, c_ells_to_bin):
        """
        Bin the power spectrum to get the binned power spectrum

        Parameters
        ----------
        c_ells_to_bin: array[float] of dimensions [lmax+1, nstokes, nstokes]
            power spectrum to bin

        Returns
        -------
        binned_c_ells: array[float] of dimensions [number_bins, nstokes, nstokes]
            Binned power spectrum
        """
        # chx.assert_axis_dimension(c_ells_to_bin, 0, self.lmax+1 - self.lmin)

        ell_distribution = jnp.arange(self.lmin, self.lmax + 1)

        def map_binned_c_ells(bin_sup, bin_inf):
            """Compute the binned power spectrum for a given ell"""
            cond_bool = jnp.logical_and(bin_inf <= ell_distribution, bin_sup > ell_distribution)
            cond_indices = jnp.where(cond_bool, 1, 0)
            return (c_ells_to_bin * cond_indices).sum() / (bin_sup - bin_inf)

        binned_c_ells = jax.vmap(map_binned_c_ells)(self.bin_ell_distribution[1:], self.bin_ell_distribution[:-1])

        chx.assert_axis_dimension(binned_c_ells, 0, self.number_bins)
        return binned_c_ells

    def bin_and_reproject_red_c_ell(self, red_cov_matrix):
        """
        Bin and reproject the power spectrum

        Parameters
        ----------
        c_ells_to_reproject: array[float] of dimensions [lmax+1, nstokes, nstokes]
            power spectrum to bin and reproject

        Returns
        -------
        binned_reprojected_c_ells: array[float] of dimensions [lmax+1, nstokes, nstokes]
            Binned and reprojected power spectrum
        """

        def map_bin_and_reproject_red_c_ell(c_ell):
            """Compute the binned and reprojected power spectrum for a given c_ell"""
            return self.project_bin_to_ell(self.get_binned_c_ells(c_ell))

        def map_bin_and_reproject(c_ell_nstokes):
            """Compute the binned and reprojected power spectrum for a given c_ell"""
            return jax.vmap(map_bin_and_reproject_red_c_ell)(c_ell_nstokes)

        return (jax.vmap(map_bin_and_reproject)(red_cov_matrix.swapaxes(0, -1))).swapaxes(0, -1)

    def get_band_limited_maps(self, input_map):
        """Get band limited maps from input maps between lmin and lmax

        Parameters
        ----------
        input_map: array[float] of dimensions [nstokes, n_pix]
            input maps to be band limited

        Returns
        -------
        array[float] of dimensions [nstokes, n_pix]
            band limited maps
        """

        covariance_unity = jnp.zeros((self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
        covariance_unity = covariance_unity.at[:, ...].set(jnp.eye(self.nstokes))
        return maps_x_red_covariance_cell_JAX(
            jnp.copy(input_map), covariance_unity, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter
        )

    def get_cond_unobserved_patches(self):
        """
        Get boolean condition on the free B_f indices corresponding to patches within the mask
        """

        templates = self.get_all_templates()
        # mask_with_m1 = jnp.where(self.mask==0, -1, 1)
        templates = templates.at[:, :, self.mask == 0].set(-1)
        return jnp.isin(jnp.arange(self.len_params), jnp.unique(templates))

    def get_cond_unobserved_patches_from_indices(self, indices):
        """
        Get boolean condition on the free B_f indices corresponding to patches within the mask
        """

        templates = self.get_all_templates()
        templates = templates.at[:, :, self.mask == 0].set(-1)
        return jnp.isin(indices, jnp.unique(templates))

    def get_sampling_eta_v2(
        self,
        red_cov_approx_matrix_sqrt,
        invBtinvNB,
        BtinvN_sqrt,
        jax_key_PNRG,
        map_random_x=None,
        map_random_y=None,
        suppress_low_modes=True,
    ):
        """Sampling step 1: eta maps
        Solve CG for eta term with formulation:
            eta = \tilde{C}^(1/2) N_c^{-1/2} x + y

        Or, fully developped:
            eta = \tilde{C}^(1/2) ( (E (B^t N^{-1} B)^{-1} E) E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x + \tilde{C}^(-1/2) y )
        This computation is performed assuming \tilde{C} (or C_approx) is block diagonal

        Parameters
        ----------
        red_cov_approx_matrix_sqrt: array[float] of dimensions [lmax+1-lmin, nstokes, nstokes]
            matrix square root of the covariance matrice (\tilde{C} or C_approx) in harmonic domain
        invBtinvNB: array[float] of dimensions [component, component, n_pix]
            matrix (B^t N^{-1} B)^{-1}
        BtinvN_sqrt: array[float] of dimensions [component, frequencies, n_pix]
            matrix B^T N^{-1/2}
        jax_key_PNRG: array[jnp.uint32]
            random key for JAX PNRG

        map_random_x: array[float] of dimensions [nfreq, nstokes, n_pix] (optional)
            set of maps 0 with mean and variance 1, which will be used to compute eta, default None (to have them computed in the routine)
        map_random_y: array[float] of dimensions [nstokes, n_pix] (optional)
            set of maps 0 with mean and variance 1, which will be used to compute eta, default None (to have them computed in the routine)

        suppress_low_modes: bool (optional)
            if True, suppress low modes in the CG between lmin and lmax, default True

        Returns
        -------
        eta_maps: array[float] of dimensions [nstokes, n_pix]
            eta maps for B_f sampling
        """

        # Chex test for arguments
        chx.assert_shape(red_cov_approx_matrix_sqrt, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
        chx.assert_shape(invBtinvNB, (self.n_components, self.n_components, self.n_pix))
        chx.assert_shape(BtinvN_sqrt, (self.n_components, self.n_frequencies, self.n_pix))

        # Creation of the random maps if they are not given
        jax_key_PNRG, jax_key_PNRG_x = random.split(jax_key_PNRG)  # Splitting of the random key to generate a new one

        if map_random_x is None:
            # If no random maps are provided, then it is computed within the routine
            print('Recalculating x !')
            map_random_x = jax.random.normal(
                jax_key_PNRG_x, shape=(self.n_frequencies, self.nstokes, self.n_pix)
            )  # /jhp.nside2resol(nside)

        jax_key_PNRG, jax_key_PNRG_y = random.split(jax_key_PNRG)  # Splitting of the random key to generate a new one

        if map_random_y is None:
            # If no random maps are provided, then it is computed within the routine
            print('Recalculating y !')
            map_random_y = jax.random.normal(jax_key_PNRG_y, shape=(self.nstokes, self.n_pix)) / jhp.nside2resol(
                self.nside
            )

        # Computation of the right hand side member of the equation

        ## First right member N_c^{-1/2} x = (E (B^t N^{-1} B)^{-1} E) E (B^t N^{-1} B)^{-1} B^t N^{-1/2} x
        N_c_inv = jnp.copy(invBtinvNB[0, 0])
        N_c_inv = N_c_inv.at[..., self.mask != 0].set(
            1 / invBtinvNB[0, 0, self.mask != 0] / jhp.nside2resol(self.nside) ** 2
        )
        first_member = (
            jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_x)[0] * N_c_inv
        )  # Selecting CMB component of the random variable

        ## Second right member is  C_approx^(1/2) C_approx^(-1/2) y, so no need to compute it

        # Getting final solution: C_approx^(1/2) ( N_c^{-1/2} x + C_approx^(-1/2) y)

        ## First getting the full first member C_approx^(1/2) N_c^{-1/2} x
        full_first_member = maps_x_red_covariance_cell_JAX(
            first_member, red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter
        )

        ## The second member is simply C_approx^(1/2) C_approx^(-1/2) y

        # Retrieving the solution
        map_solution = map_random_y + full_first_member

        # If needed, making it band-limited
        if suppress_low_modes:
            map_solution = self.get_band_limited_maps(map_solution)
        return map_solution

    def get_fluctuating_term_maps_v2d(
        self,
        red_cov_matrix_sqrt,
        invBtinvNB,
        BtinvN_sqrt,
        jax_key_PNRG,
        map_random_realization_xi=None,
        map_random_realization_chi=None,
        initial_guess=jnp.empty(0),
        precond_func=None,
    ):
        r"""Sampling step 2: fluctuating term

        Solve fluctuation term:
            (Id + C^{1/2} N_c^{-1} C^{1/2}) C^{-1/2} \zeta = xi + C^{1/2} N_c^{-1/2} \chi

        Or, fully developped:
            (Id + C^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C^{1/2}) \zeta = C^{1/2} C^{-1/2} xi + C^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi

        This ensures:
        <\zeta \zeta^t> = (C^{-1} + N_c^{-1})^{-1}
        < \zeta > = 0

        Note C is assumed to be block diagonal

        Parameters
        ----------
        red_cov_matrix_sqrt: array[float] of dimension [lmin:lmax, nstokes, nstokes]
            term C^{1/2}, matrix square root of CMB covariance matrices in harmonic domain
        invBtinvNB: array[float] of dimension [component, component, n_pix]
            matrix (B^t N^{-1} B)^{-1}
        BtinvN_sqrt: array[float] of dimension [component, frequencies, n_pix]
            matrix B^T N^{-1/2}
        jax_key_PNRG: array[jnp.uint32]
            random key for JAX PNRG
        map_white_noise_xi: array[float] of dimensions [nstokes, n_pix] (optional)
            set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term
        map_white_noise_chi: array[float] of dimensions [nfreq, nstokes, n_pix] (optional)
            set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term
        initial_guess: array[float] of dimensions [nstokes, n_pix] (optional)
            initial guess for the CG, default jnp.empty(0) (then set to 0)
        precond_func: function (optional)
            function preconditioner for the CG, default None

        Returns
        -------
        fluctuating_map_z: array[float] of dimensions [nstokes, n_pix]
            Fluctuation maps for s_c sampling
        """

        # Chex test for arguments
        chx.assert_axis_dimension(red_cov_matrix_sqrt, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(invBtinvNB, 2, self.n_pix)
        chx.assert_axis_dimension(BtinvN_sqrt, 1, self.n_frequencies)
        chx.assert_axis_dimension(BtinvN_sqrt, 2, self.n_pix)

        jax_key_PNRG, jax_key_PNRG_xi = random.split(jax_key_PNRG)  # Splitting of the random key to generate a new one

        # Creation of the random maps if they are not given
        if map_random_realization_xi is None:
            # If no random maps are provided, then it is computed within the routine
            print('Recalculating xi !')
            map_random_realization_xi = jax.random.normal(
                jax_key_PNRG_xi, shape=(self.nstokes, self.n_pix)
            ) / jhp.nside2resol(self.nside)

        jax_key_PNRG, *jax_key_PNRG_chi = random.split(
            jax_key_PNRG, self.n_frequencies + 1
        )  # Splitting of the random key to generate a new one
        if map_random_realization_chi is None:
            # If no random maps are provided, then it is computed within the routine
            print('Recalculating chi !')

            def fmap(random_key):
                random_map = jax.random.normal(random_key, shape=(self.nstokes, self.n_pix))
                # return self.get_band_limited_maps(random_map)
                return random_map

            map_random_realization_chi = jax.vmap(fmap)(
                jnp.array(jax_key_PNRG_chi)
            )  # Generating a different random Gaussian map for each frequency

            chx.assert_shape(map_random_realization_chi, (self.n_frequencies, self.nstokes, self.n_pix))

        # Computation of the right side member of the CG

        # First right member: xi
        right_member_1 = map_random_realization_xi

        # Second right member:
        ## Computation of C^{1/2} N_c^{-1/2} \chi
        N_c_inv = jnp.copy(invBtinvNB[0, 0])
        N_c_inv = N_c_inv.at[..., self.mask != 0].set(
            1 / invBtinvNB[0, 0, self.mask != 0] / jhp.nside2resol(self.nside) ** 2
        )
        N_c_inv_repeat = jnp.broadcast_to(N_c_inv, (self.nstokes, self.n_pix)).ravel()
        # Repeat N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Computation of N_c^{-1/2} \chi = (E^t (B^t N^{-1} B)^{-1} E) E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi
        right_member_2_part = (
            jnp.einsum('kcp,cfp,fsp->ksp', invBtinvNB, BtinvN_sqrt, map_random_realization_chi)[0] * N_c_inv
        )  # [0] for selecting CMB component of the random variable
        # First compute N_c^{-1/2} \chi
        right_member_2 = maps_x_red_covariance_cell_JAX(
            right_member_2_part, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter
        )
        # Then apply C^{1/2} to N_c^{-1/2} \chi

        # right_member = (right_member_1 + right_member_2).ravel()
        right_member = self.get_band_limited_maps(right_member_1).ravel() + right_member_2.ravel()

        # Computation of the left side member of the equation

        # Operator in harmonic domain: C^{1/2}
        first_part_term_left = lambda x: maps_x_red_covariance_cell_JAX(
            x.reshape((self.nstokes, self.n_pix)),
            red_cov_matrix_sqrt,
            nside=self.nside,
            lmin=self.lmin,
            n_iter=self.n_iter,
        ).ravel()

        ## Operator in pixel domain: (E^t (B^t N^{-1} B) E)^{-1}
        def second_part_term_left(x):
            return x * N_c_inv_repeat

        ## Defining the function to inverse with the CG
        func_left_term = lambda x: x.ravel() + first_part_term_left(second_part_term_left(first_part_term_left(x)))

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(map_random_realization_xi)

        # Actual start of the CG

        # Start of the CG
        time_start = time.time()
        fluctuating_map_z, number_iterations = jsp.sparse.linalg.cg(
            func_left_term,
            right_member.ravel(),
            x0=initial_guess.ravel(),
            tol=self.tolerance_CG,
            maxiter=self.limit_iter_cg,
            M=precond_func,
        )
        ## Computing the term C^{-1/2} \zeta

        print('CG Fluct finished with', number_iterations, 'iterations in ', time.time() - time_start, 'seconds !!')

        fluctuating_map = maps_x_red_covariance_cell_JAX(
            fluctuating_map_z.reshape((self.nstokes, self.n_pix)),
            red_cov_matrix_sqrt,
            nside=self.nside,
            lmin=self.lmin,
            n_iter=self.n_iter,
        )

        ## Retrieving \zeta from C^{-1/2} \zeta
        return fluctuating_map.reshape((self.nstokes, self.n_pix))

    def solve_generalized_wiener_filter_term_v2d(
        self, s_cML, red_cov_matrix_sqrt, invBtinvNB, initial_guess=jnp.empty(0), precond_func=None
    ):
        """
        Solve Wiener filter term with CG:
             (Id + C^{1/2} N_c^{-1} C^{1/2}) C^{-1/2} s_{c,WF} = C^{1/2} N_c^{-1} s_{c,ML}

        This ensures:
            s_{c,WF} = (C^{-1} + N_c^{-1})^{-1} N_c^{-1} s_{c,ML}
        Note C is assumed to be block diagonal

        Parameters
        ----------
        s_cML: array[float] of dimensions [nstokes, n_pix]
            Maximum Likelihood solution of component separation from input frequency maps
        red_cov_matrix_sqrt: array[float] of dimension [lmin:lmax, nstokes, nstokes]
            term C^{1/2}, matrix square root of CMB covariance matrices in harmonic domain
        invBtinvNB: array[float] of dimension [component, component, n_pix]
            matrix (B^t N^{-1} B)^{-1}
        initial_guess: array[float] of dimensions [nstokes, n_pix] (optional)
            initial guess for the CG, default jnp.empty(0) (then set to 0)
        precond_func: function (optional)
            function preconditioner for the CG, default None

        Returns
        -------
        array[float] of dimensions [nstokes, n_pix]
            Wiener filter maps for s_c sampling
        """

        # Chex test for arguments
        chx.assert_shape(red_cov_matrix_sqrt, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
        chx.assert_shape(s_cML, (self.nstokes, self.n_pix))
        chx.assert_shape(invBtinvNB, (self.n_components, self.n_components, self.n_pix))

        # Computation of the right side member of the CG: C^{1/2} N_c^{-1} s_c,ML
        ## First, computation of N_c^{-1} taking into account the mask
        N_c_repeat = jnp.broadcast_to(
            invBtinvNB[0, 0] * jhp.nside2resol(self.nside) ** 2, (self.nstokes, self.n_pix)
        ).ravel()
        ## Repeat N_c for each Stokes parameter, for speed-up afterwards
        N_c_inv = jnp.copy(invBtinvNB[0, 0])
        N_c_inv = N_c_inv.at[..., self.mask != 0].set(
            1 / invBtinvNB[0, 0, self.mask != 0] / jhp.nside2resol(self.nside) ** 2
        )
        N_c_inv_repeat = jnp.broadcast_to(
            N_c_inv, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Then, computation of C^{1/2} N_c^{-1} s_c,ML
        right_member = maps_x_red_covariance_cell_JAX(
            s_cML * N_c_inv, red_cov_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter
        ).ravel()

        # Preparation of the harmonic operator C^{1/2} for the LHS of the CG
        first_part_term_left = lambda x: maps_x_red_covariance_cell_JAX(
            x.reshape((self.nstokes, self.n_pix)),
            red_cov_matrix_sqrt,
            nside=self.nside,
            lmin=self.lmin,
            n_iter=self.n_iter,
        ).ravel()

        ## Second left member pixel operator: (E^t (B^t N^{-1} B)^{-1} E) x
        def second_part_term_left(x):
            return x * N_c_inv_repeat

        # Full operator to inverse: Id + C^{1/2} N_c^{-1} C^{1/2}
        func_left_term = lambda x: x.ravel() + first_part_term_left(second_part_term_left(first_part_term_left(x)))

        # Initial guess for the CG
        if jnp.size(initial_guess) == 0:
            initial_guess = jnp.zeros_like(s_cML)

        # Actual start of the CG

        time_start = time.time()
        wiener_filter_term_z, number_iterations = jsp.sparse.linalg.cg(
            func_left_term,
            right_member.ravel(),
            x0=initial_guess.ravel(),
            tol=self.tolerance_CG,
            maxiter=self.limit_iter_cg,
            M=precond_func,
        )  # atol=self.atol_CG,
        ## Computing the term C^{-1/2} s_{c,WF}

        print('CG WF finished with', number_iterations, 'iterations in ', time.time() - time_start, 'seconds !!')

        wiener_filter_term = maps_x_red_covariance_cell_JAX(
            wiener_filter_term_z.reshape((self.nstokes, self.n_pix)),
            red_cov_matrix_sqrt,
            nside=self.nside,
            lmin=self.lmin,
            n_iter=self.n_iter,
        )
        ## Retrieving s_{c,WF} from C^{-1/2} s_{c,WF}

        return wiener_filter_term.reshape((self.nstokes, self.n_pix))

    def get_inverse_wishart_sampling_from_c_ells(self, sigma_ell, PRNGKey, old_sample=None, acceptance_posdef=False):
        """
        Solve sampling step 3: inverse Wishart distribution with C

        sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

        Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000):
            1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
            2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
            3. Invert this matrix to obtain the final result: a matrix sample following an inverse Wishart distribution

        Also assumes the monopole and dipole to be 0

        Parameters
        ----------
        sigma_ell: array[float] of dimensions [n_correlations, lmin:lmax+1]
            initial power spectrum which will define the parameter matrix of the inverse Wishart distribution
        PRNGKey: array[jnp.uint32]
            random key for JAX PNRG

        Returns
        -------
        array[float] of dimensions [lmin:lmax+1, nstokes, nstokes]
            Matrices following an inverse Wishart distribution
        """

        chx.assert_axis_dimension(sigma_ell, 1, self.lmax + 1 - self.lmin)
        c_ells_Wishart_modified = jnp.copy(sigma_ell) * (2 * jnp.arange(self.lmin, self.lmax + 1) + 1)
        invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))

        sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)

        def map_sampling_Wishart(ell_PNRGKey, ell):
            """Compute the sampling of the Wishart distribution for a given ell"""

            sample_gaussian = random.multivariate_normal(
                ell_PNRGKey,
                jnp.zeros(self.nstokes),
                invert_parameter_Wishart[ell],
                shape=(2 * self.lmax - self.nstokes,),
            )

            weighting = jnp.where(ell >= (jnp.arange(2 * self.lmax - self.nstokes) + self.nstokes) / 2, 1, 0)

            sample_to_return = jnp.einsum('lk,l,lm->km', sample_gaussian, weighting, sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return

        PRNGKey_map = random.split(PRNGKey, self.lmax - self.lmin + 1)  # Prepare lmax+1-lmin PRNGKeys to be used
        # sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        sampling_Wishart = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin, self.lmax + 1))
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        # sampling_Wishart = sampling_Wishart.at[self.lmin:].set(sampling_Wishart_map)
        # return jnp.linalg.pinv(sampling_Wishart)
        reconstructed_spectra = jnp.linalg.pinv(sampling_Wishart)
        new_sample = reconstructed_spectra
        if acceptance_posdef and old_sample is not None:
            print('Only positive definite matrices are accepted for inv Wishart !')
            eigen_prod = jnp.prod(jnp.linalg.eigvalsh(reconstructed_spectra), axis=(1))
            acceptance = jnp.where(eigen_prod < 0, 0, 1)
            acceptance_reversed = jnp.where(eigen_prod < 0, 1, 0)
            # new_sample = jnp.copy(old_sample)
            # new_sample = new_sample.at[acceptance==1,...].set(reconstructed_spectra[acceptance==1,...])
            new_sample = jnp.einsum('lik,l->lik', reconstructed_spectra, acceptance) + jnp.einsum(
                'lik,l->lik', old_sample, acceptance_reversed
            )
        return new_sample

    def get_conditional_proba_C_from_previous_sample(self, red_sigma_ell, red_cov_matrix_sampled):
        """Compute log-proba of C. The associated log proba is:
            -1/2 (tr sigma_ell C^-1) - 1/2 log det C

        Parameters
        ----------
        red_sigma_ell: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            covariance matrices in harmonic domain
        red_cov_matrix_sampled: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            previous sampled covariance matrices in harmonic domain
        Returns
        -------
        float
            log-proba of C parametrized by r_param
        """
        chx.assert_equal_shape((red_sigma_ell, red_cov_matrix_sampled))

        # Getting determinant of the covariance matrix
        sum_dets = (
            (2 * jnp.arange(self.lmin, self.lmax + 1) + 1) * jnp.log(jnp.abs(jnp.linalg.det(red_cov_matrix_sampled)))
        ).sum()

        return -(jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets) / 2

    def get_inverse_gamma_sampling_from_c_ells(self, sigma_ell, PRNGKey):
        """Solve sampling step 3: inverse Gamma distribution with C

        sigma_ell is expected to be exactly the parameter of the inverse Gamma (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

        Also assumes the monopole and dipole to be 0

        Parameters
        ----------
        sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Gamma distribution ; must be of dimension [n_correlations, lmax+1]
        PRNGKey: random key for JAX PNRG

        Returns
        -------
        Matrices following an inverse Gamma distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        chx.assert_axis_dimension(sigma_ell, 1, self.lmax + 1)

        # c_ells_gamma_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)

        c_ells_gamma_modified = jnp.zeros_like(sigma_ell)
        c_ells_gamma_modified = c_ells_gamma_modified.at[: self.nstokes].set(
            jnp.copy(sigma_ell[: self.nstokes]) * (2 * jnp.arange(self.lmax + 1) + 1)
        )
        red_c_ell_Gamma_b_factor = get_reduced_matrix_from_c_ell_jax(c_ells_gamma_modified) / 2

        sampling_Gamma = jnp.zeros_like(red_c_ell_Gamma_b_factor)

        def map_sampling_Gamma(ell_PNRGKey, ell):
            """Compute the sampling of the Gamma distribution for a given ell"""
            return jnp.diag(
                jnp.diag(red_c_ell_Gamma_b_factor[ell]) / random.gamma(ell_PNRGKey, a=(2 * ell + 1 - 2) / 2, shape=(1,))
            )

        PRNGKey_map = random.split(PRNGKey, self.lmax - self.lmin + 1)  # Prepare lmax+1-lmin PRNGKeys to be used
        sampling_Gamma_map = jax.vmap(map_sampling_Gamma)(PRNGKey_map, jnp.arange(self.lmin, self.lmax + 1))
        # Map over PRNGKeys and ells to create samples of the Gamma distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Gamma = sampling_Gamma.at[self.lmin :].set(sampling_Gamma_map)
        return sampling_Gamma

    def get_binned_red_c_ells_v2(self, red_c_ells_to_bin):
        """Bin the power spectrum to get the binned power spectrum

        Parameters
        ----------
        red_c_ells_to_bin: power spectrum to bin ; must be of dimension [lmax+1, nstokes, nstokes]

        Returns
        -------
        Binned power spectrum, of dimension [number_bins, nstokes, nstokes]
        """
        chx.assert_axis_dimension(red_c_ells_to_bin, 0, self.lmax + 1 - self.lmin)

        ell_distribution = jnp.arange(red_c_ells_to_bin.shape[0]) + self.lmin

        # number_bins = self.bin_ell_distribution.shape[0]-1

        def map_binned_red_c_ells(bin_ell):
            """Compute the binned power spectrum for a given ell"""
            cond = jnp.logical_and(
                self.bin_ell_distribution[bin_ell] <= ell_distribution,
                self.bin_ell_distribution[bin_ell + 1] > ell_distribution,
            )
            cond_quantitative = jnp.where(cond, 1, 0)
            modes_to_bin = jnp.einsum(
                'lij, l, l->lij', red_c_ells_to_bin, cond_quantitative, ell_distribution * (ell_distribution + 1)
            )

            return modes_to_bin.sum(
                axis=0
            )  # /(self.bin_ell_distribution[bin_ell+1]-self.bin_ell_distribution[bin_ell])

        binned_red_c_ells = jax.vmap(map_binned_red_c_ells)(jnp.arange(self.number_bins))
        return binned_red_c_ells

    def get_binned_inverse_wishart_sampling_from_c_ells_v2(
        self, sigma_ell, PRNGKey, old_sample=None, acceptance_posdef=False
    ):
        """Solve sampling step 3: inverse Wishart distribution with C

        sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

        Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000):
            1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
            2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
            3. Invert this matrix to obtain the final result: a matrix sample following an inverse Wishart distribution

        Also assumes the monopole and dipole to be 0

        Parameters
        ----------
        sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [n_correlations, lmax+1]
        PRNGKey: random key for JAX PNRG

        Returns
        -------
        Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        # chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)
        chx.assert_axis_dimension(sigma_ell, 1, self.lmax + 1 - self.lmin)
        # c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        c_ells_Wishart_modified = jnp.copy(sigma_ell) * (2 * (jnp.arange(self.lmax + 1 - self.lmin) + self.lmin) + 1)
        # invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        binned_invert_parameter_Wishart = jnp.linalg.pinv(
            self.get_binned_red_c_ells_v2(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        )

        # sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)
        sampling_Wishart = jnp.zeros_like(binned_invert_parameter_Wishart)
        # number_dof = lambda x: 2*x+1

        def map_sampling_Wishart(ell_PNRGKey, binned_ell):
            """Compute the sampling of the Wishart distribution for a given ell"""

            sample_gaussian = random.multivariate_normal(
                ell_PNRGKey,
                jnp.zeros(self.nstokes),
                binned_invert_parameter_Wishart[binned_ell],
                shape=(self.maximum_number_dof - self.nstokes - 1,),
            )

            weighting = jnp.where(
                self.number_dof(binned_ell) - self.nstokes - 1
                >= jnp.arange(self.maximum_number_dof - self.nstokes - 1),
                1,
                0,
            )

            sample_to_return = jnp.einsum('lk,l,lm->km', sample_gaussian, weighting, sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return

        # distribution_bins_over_lmin = self.bin_ell_distributed_min[self.bin_ell_distributed_min>=self.lmin]
        distribution_bins_over_lmin = jnp.copy(self.bin_ell_distribution)[:-1]

        # number_bins_over_lmin = distribution_bins_over_lmin.shape[0]
        # number_bins_over_lmin = jnp.size(distribution_bins_over_lmin)

        # PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        PRNGKey_map = random.split(PRNGKey, self.number_bins)  # Prepare lmax+1-lmin PRNGKeys to be used

        # sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.number_bins))
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Wishart = sampling_Wishart.at[-self.number_bins :].set(sampling_Wishart_map)
        sampling_invWishart = jnp.linalg.pinv(sampling_Wishart)

        def reconstruct_spectra(ell):
            cond = jnp.logical_and(self.bin_ell_distribution[:-1] <= ell, self.bin_ell_distribution[1:] > ell)
            bin_contribution = jnp.where(cond, 1, 0)
            return jnp.einsum('lij,l->ij', sampling_invWishart, bin_contribution) / (ell * (ell + 1))

        # reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.lmin,self.lmax+1))
        reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.lmin, self.bin_ell_distribution[-1]))

        if acceptance_posdef and old_sample is not None:
            print('Only positive definite matrices are accepted for inv Wishart !')
            eigen_prod = jnp.prod(jnp.linalg.eigvalsh(reconstructed_spectra), axis=(1))
            acceptance = jnp.where(eigen_prod < 0, 0, 1)
            acceptance_reversed = jnp.where(eigen_prod < 0, 1, 0)
            # new_sample = jnp.copy(old_sample)
            # new_sample = new_sample.at[acceptance==1,...].set(reconstructed_spectra[acceptance==1,...])
            new_sample = jnp.einsum('lik,l->lik', reconstructed_spectra, acceptance) + jnp.einsum(
                'lik,l->lik', old_sample, acceptance_reversed
            )

        chx.assert_tree_all_finite(new_sample)
        return new_sample
        # return reconstructed_spectra

    def get_binned_red_c_ells_v3(self, red_c_ells_to_bin):
        """Bin the power spectrum to get the binned power spectrum

        Parameters
        ----------
        red_c_ells_to_bin: power spectrum to bin ; must be of dimension [lmax+1, nstokes, nstokes]

        Returns
        -------
        Binned power spectrum, of dimension [number_bins, nstokes, nstokes]
        """
        chx.assert_axis_dimension(red_c_ells_to_bin, 0, self.lmax + 1 - self.lmin)

        ell_distribution = jnp.arange(red_c_ells_to_bin.shape[0]) + self.lmin

        # number_bins = self.bin_ell_distribution.shape[0]-1

        def map_binned_red_c_ells(bin_ell):
            """Compute the binned power spectrum for a given ell"""
            cond = jnp.logical_and(
                self.bin_ell_distribution[bin_ell] <= ell_distribution,
                self.bin_ell_distribution[bin_ell + 1] > ell_distribution,
            )
            cond_quantitative = jnp.where(cond, 1, 0)
            modes_to_bin = jnp.einsum('lij, l->lij', red_c_ells_to_bin, cond_quantitative)

            return modes_to_bin.sum(
                axis=0
            )  # /(self.bin_ell_distribution[bin_ell+1]-self.bin_ell_distribution[bin_ell])

        binned_red_c_ells = jax.vmap(map_binned_red_c_ells)(jnp.arange(self.number_bins))
        return binned_red_c_ells

    def get_binned_red_c_ells_v4(self, red_c_ells_to_bin):
        """Bin the power spectrum to get the binned power spectrum

        Parameters
        ----------
        red_c_ells_to_bin: power spectrum to bin ; must be of dimension [lmax+1, nstokes, nstokes]

        Returns
        -------
        Binned power spectrum, of dimension [number_bins, nstokes, nstokes]
        """
        chx.assert_axis_dimension(red_c_ells_to_bin, 0, self.lmax + 1 - self.lmin)

        ell_distribution = jnp.arange(red_c_ells_to_bin.shape[0]) + self.lmin

        # number_bins = self.bin_ell_distribution.shape[0]-1

        def map_binned_red_c_ells(bin_sup, bin_inf):
            """Compute the binned power spectrum for a given ell"""
            cond = jnp.logical_and(bin_inf <= ell_distribution, bin_sup > ell_distribution)
            cond_quantitative = jnp.where(cond, 1, 0)
            modes_to_bin = jnp.einsum('lij, l->lij', red_c_ells_to_bin, cond_quantitative)

            return modes_to_bin.sum(
                axis=0
            )  # /(self.bin_ell_distribution[bin_ell+1]-self.bin_ell_distribution[bin_ell])

        binned_red_c_ells = jax.vmap(map_binned_red_c_ells)(
            self.bin_ell_distribution[1:], self.bin_ell_distribution[:-1]
        )

        chx.assert_axis_dimension(binned_red_c_ells, 0, self.number_bins)
        return binned_red_c_ells

    def get_binned_inverse_wishart_sampling_from_c_ells_v3(
        self, sigma_ell, PRNGKey, old_sample=None, acceptance_posdef=False
    ):
        """Solve sampling step 3: inverse Wishart distribution with C

        sigma_ell is expected to be exactly the parameter of the inverse Wishart (so it should NOT be multiplied by 2*ell+1 if it is thought as a power spectrum)

        Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000):
            1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
            2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
            3. Invert this matrix to obtain the final result: a matrix sample following an inverse Wishart distribution

        Also assumes the monopole and dipole to be 0

        Parameters
        ----------
        sigma_ell: initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [n_correlations, lmax+1]
        PRNGKey: random key for JAX PNRG

        Returns
        -------
        Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
        """

        # chx.assert_axis_dimension(sigma_ell, 1, self.lmax+1)
        chx.assert_axis_dimension(sigma_ell, 1, self.lmax + 1 - self.lmin)
        # c_ells_Wishart_modified = jnp.copy(sigma_ell)*(2*jnp.arange(self.lmax+1) + 1)
        c_ells_Wishart_modified = (sigma_ell * (2 * jnp.arange(self.lmin, self.lmax + 1) + 1))[
            ..., : self.bin_ell_distribution[-1] - self.lmin
        ]  # -1 ?

        chx.assert_axis_dimension(c_ells_Wishart_modified, 1, self.bin_ell_distribution[-1] - self.lmin)  # -1 ?

        # invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        # binned_invert_parameter_Wishart = jnp.linalg.pinv(self.get_binned_red_c_ells_v3(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)))
        binned_invert_parameter_Wishart = jnp.linalg.pinv(
            self.get_binned_red_c_ells_v4(get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified))
        )

        # sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)
        sampling_Wishart = jnp.zeros_like(binned_invert_parameter_Wishart)
        # number_dof = lambda x: 2*x+1

        all_number_dof = jnp.array((self.bin_ell_distribution[1:]) ** 2 - self.bin_ell_distribution[:-1] ** 2)

        def map_sampling_Wishart(ell_PNRGKey, binned_ell):
            """Compute the sampling of the Wishart distribution for a given ell"""

            sample_gaussian = random.multivariate_normal(
                ell_PNRGKey,
                jnp.zeros(self.nstokes),
                binned_invert_parameter_Wishart[binned_ell],
                shape=(self.maximum_number_dof - self.nstokes - 1,),
            )

            # weighting = jnp.where(self.number_dof(binned_ell)-self.nstokes-1 >= jnp.arange(self.maximum_number_dof-self.nstokes-1), 1, 0)
            weighting = jnp.where(
                all_number_dof[binned_ell] - self.nstokes - 1 >= jnp.arange(self.maximum_number_dof - self.nstokes - 1),
                1,
                0,
            )

            sample_to_return = jnp.einsum('lk,l,lm->km', sample_gaussian, weighting, sample_gaussian)
            # new_carry = new_ell_PRNGKey
            return sample_to_return

        # distribution_bins_over_lmin = self.bin_ell_distributed_min[self.bin_ell_distributed_min>=self.lmin]
        distribution_bins_over_lmin = jnp.copy(self.bin_ell_distribution)[:-1]

        # number_bins_over_lmin = distribution_bins_over_lmin.shape[0]
        # number_bins_over_lmin = jnp.size(distribution_bins_over_lmin)

        # PRNGKey_map = random.split(PRNGKey, self.lmax-self.lmin+1) # Prepare lmax+1-lmin PRNGKeys to be used
        PRNGKey_map = random.split(PRNGKey, self.number_bins)  # Prepare lmax+1-lmin PRNGKeys to be used

        # sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.lmin,self.lmax+1))
        sampling_Wishart_map = jax.vmap(map_sampling_Wishart)(PRNGKey_map, jnp.arange(self.number_bins))
        # Map over PRNGKeys and ells to create samples of the Wishart distribution, of dimension [lmax+1-lmin,nstokes,nstokes]

        sampling_Wishart = sampling_Wishart.at[:].set(sampling_Wishart_map)
        sampling_invWishart = jnp.linalg.pinv(sampling_Wishart)

        def reconstruct_spectra(ell):
            cond = jnp.logical_and(self.bin_ell_distribution[:-1] <= ell, self.bin_ell_distribution[1:] > ell)
            bin_contribution = jnp.where(cond, 1, 0)
            # return jnp.einsum('lij,l->ij',sampling_invWishart,bin_contribution)#/(ell*(ell+1))
            return jnp.einsum('lij,l->ij', sampling_invWishart, bin_contribution) / bin_contribution.sum()

        # reconstructed_spectra = jax.vmap(reconstruct_spectra)(jnp.arange(self.lmin,self.lmax+1))
        reconstructed_spectra = jax.vmap(reconstruct_spectra)(
            jnp.arange(self.bin_ell_distribution[0], self.bin_ell_distribution[-1]) - 1
        )

        new_sample = reconstructed_spectra
        if acceptance_posdef and old_sample is not None:
            print('Only positive definite matrices are accepted for inv Wishart !')
            eigen_prod = jnp.prod(jnp.linalg.eigvalsh(reconstructed_spectra), axis=(1))
            acceptance = jnp.where(eigen_prod <= 0, 0, 1)
            acceptance_reversed = jnp.where(eigen_prod <= 0, 1, 0)
            # new_sample = jnp.copy(old_sample)
            # new_sample = new_sample.at[acceptance==1,...].set(reconstructed_spectra[acceptance==1,...])
            new_sample = jnp.einsum('lik,l->lik', reconstructed_spectra, acceptance) + jnp.einsum(
                'lik,l->lik', old_sample, acceptance_reversed
            )

        # chx.assert_tree_all_finite(new_sample)
        return new_sample

    # return reconstructed_spectra

    def get_conditional_proba_C_from_r(
        self, r_param, red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total
    ):
        """
        Compute log-proba of C parametrized by r_param.
        The parametrisation is given by: C(r) = r * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        The associated log proba is:
            -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

        Parameters
        ----------
        r_param: float
            parameter of the covariance C to be sampled
        red_sigma_ell: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            covariance matrices in harmonic domain
        theoretical_red_cov_r1_tensor: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            tensor mode covariance matrices in harmonic domain
        theoretical_red_cov_r0_total: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            scalar mode covariance matrices in harmonic domain

        Returns
        -------
        float
            log-proba of C parametrized by r_param
        """

        chx.assert_shape(theoretical_red_cov_r1_tensor, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
        chx.assert_equal_shape((red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total))

        # Getting the sigma_ell in the format [lmax,nstokes,nstokes] multiplied by 2 ell+1, to take into account the m
        red_sigma_ell = jnp.einsum('lij,l->lij', red_sigma_ell, 2 * jnp.arange(self.lmin, self.lmax + 1) + 1)

        # Getting the CMB covariance matrix parametrized by r_param
        red_cov_matrix_sampled = r_param * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        # Getting determinant of the covariance matrix log det C(r) ; taking into account the factor 2ell+1 for the multiples m
        sum_dets = (
            (2 * jnp.arange(self.lmin, self.lmax + 1) + 1) * jnp.log(jnp.abs(jnp.linalg.det(red_cov_matrix_sampled)))
        ).sum()

        return (
            -(jnp.einsum('lij,lji->l', red_sigma_ell, jnp.linalg.pinv(red_cov_matrix_sampled)).sum() + sum_dets) / 2
        )  # -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

    def get_conditional_proba_C_from_r_wBB(
        self, r_param, red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total
    ):
        """
        Compute log-proba of C parametrized by r_param only from the BB power spectrum.
        The parametrisation is given by: C(r) = r * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        The associated log proba is:
            -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

        Parameters
        ----------
        r_param: float
            parameter of the covariance C to be sampled
        red_sigma_ell: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            covariance matrices in harmonic domain
        theoretical_red_cov_r1_tensor: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            tensor mode covariance matrices in harmonic domain
        theoretical_red_cov_r0_total: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            scalar mode covariance matrices in harmonic domain

        Returns
        -------
        float
            log-proba of C parametrized by r_param
        """

        chx.assert_shape(theoretical_red_cov_r1_tensor, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
        chx.assert_equal_shape((red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total))

        # Getting the sigma_ell in the format [lmax,nstokes,nstokes] multiplied by 2 ell+1, to take into account the m
        red_sigma_ell = jnp.einsum('lij,l->lij', red_sigma_ell, 2 * jnp.arange(self.lmin, self.lmax + 1) + 1)

        # Getting the CMB covariance matrix parametrized by r_param
        BB_cov_matrix_sampled = r_param * theoretical_red_cov_r1_tensor[:, 1, 1] + theoretical_red_cov_r0_total[:, 1, 1]

        # Getting determinant of the covariance matrix log det C(r) ; taking into account the factor 2ell+1 for the multiples m
        sum_dets = ((2 * jnp.arange(self.lmin, self.lmax + 1) + 1) * jnp.log(jnp.abs(BB_cov_matrix_sampled))).sum()

        return (
            -((red_sigma_ell[:, 1, 1] / BB_cov_matrix_sampled).sum() + sum_dets) / 2
        )  # -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

    def get_binned_conditional_proba_C_from_r_wBB(
        self, r_param, red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total
    ):
        """
        Compute log-proba of C parametrized by r_param only from the BB power spectrum with binning.
        The parametrisation is given by: C(r) = r * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        The associated log proba is:
            -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

        Parameters
        ----------
        r_param: float
            parameter of the covariance C to be sampled
        red_sigma_ell: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            covariance matrices in harmonic domain
        theoretical_red_cov_r1_tensor: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            tensor mode covariance matrices in harmonic domain
        theoretical_red_cov_r0_total: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            scalar mode covariance matrices in harmonic domain

        Returns
        -------
        float
            log-proba of C parametrized by r_param
        """

        chx.assert_shape(theoretical_red_cov_r1_tensor, (self.lmax + 1 - self.lmin,))
        chx.assert_equal_shape((red_sigma_ell, theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total))

        # Getting the CMB covariance matrix parametrized by r_param
        BB_cov_matrix_sampled = r_param * theoretical_red_cov_r1_tensor[:, 1, 1] + theoretical_red_cov_r0_total[:, 1, 1]

        binned_BB_cov_matrix_sampled = self.get_binned_c_ells(BB_cov_matrix_sampled)

        number_dof = self.bin_ell_distribution[1:] ** 2 - self.bin_ell_distribution[:-1] ** 2

        # Getting determinant of the covariance matrix log det C(r) ; taking into account the factor 2ell+1 for the multiples m
        # sum_dets = ( (2*jnp.arange(self.lmin, self.lmax+1) +1) * jnp.log(BB_cov_matrix_sampled) ).sum()
        sum_dets = (number_dof * jnp.log(binned_BB_cov_matrix_sampled)).sum()

        # reweighted_sigma_ell = number_dof*self.get_binned_c_ells(red_sigma_ell[:,1,1]/(2*jnp.arange(self.lmin, self.lmax+1)+1))
        reweighted_sigma_ell = number_dof * self.get_binned_c_ells(red_sigma_ell[:, 1, 1])
        return (
            -((reweighted_sigma_ell / binned_BB_cov_matrix_sampled).sum() + sum_dets) / 2
        )  # -1/2 (tr sigma_ell C(r)^-1) - 1/2 log det C(r)

    def get_log_proba_non_centered_move_C(
        self,
        new_red_cov_matrix_sampled,
        old_red_cov_matrix_sampled,
        invBtinvNB,
        s_cML,
        s_c_sample,
        theoretical_red_cov_r1_tensor,
        theoretical_red_cov_r0_total,
    ):
        """
        WARNING: Exploratory, to test further
        """

        new_red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(new_red_cov_matrix_sampled)
        old_red_cov_matrix_msqrt = jnp.linalg.pinv(get_sqrt_reduced_matrix_from_matrix_jax(old_red_cov_matrix_sampled))

        N_c_inv = jnp.zeros_like(invBtinvNB[0, 0])
        N_c_inv = N_c_inv.at[..., self.mask != 0].set(
            1 / invBtinvNB[0, 0, self.mask != 0] / jhp.nside2resol(self.nside) ** 2
        )

        operator_harmonic = jnp.einsum('lij,ljk->lik', new_red_cov_matrix_sqrt, old_red_cov_matrix_msqrt)

        modified_s_c_sample = maps_x_red_covariance_cell_JAX(
            s_c_sample, operator_harmonic, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter
        )

        maps_to_compute = s_cML - modified_s_c_sample

        return (
            -jnp.einsum('sp, p, sp', maps_to_compute, N_c_inv, maps_to_compute) / 2 * jhp.nside2resol(self.nside) ** 2
        )

    def get_log_proba_non_centered_move_C_from_r(
        self,
        new_r_sample,
        old_r_sample,
        invBtinvNB,
        s_cML,
        s_c_sample,
        theoretical_red_cov_r1_tensor,
        theoretical_red_cov_r0_total,
    ):
        """
        WARNING: Exploratory, to test further
        """

        chx.assert_shape(theoretical_red_cov_r1_tensor, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
        chx.assert_equal_shape((theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total))

        # Getting the CMB covariance matrix parametrized by r_param
        new_red_cov_matrix_sampled = new_r_sample * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total
        old_red_cov_matrix_sampled = old_r_sample * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        return self.get_log_proba_non_centered_move_C(
            new_red_cov_matrix_sampled,
            old_red_cov_matrix_sampled,
            invBtinvNB,
            s_cML,
            s_c_sample,
            theoretical_red_cov_r1_tensor,
            theoretical_red_cov_r0_total,
        )

    def get_conditional_proba_spectral_likelihood_JAX(self, complete_mixing_matrix, full_data_without_CMB):
        """
        Get conditional probability of spectral likelihood from the full mixing matrix

        The associated conditional probability is given by:
        - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

        with d = full_data_without_CMB, B_c = complete_mixing_matrix, B_f = complete_mixing_matrix[:,1:,:]
        d is assumed to be band-limited

        Parameters
        ----------
        complete_mixing_matrix: array[float] of dimensions [component, frequencies]
            complete mixing matrix
        full_data_without_CMB: array[float] of dimensions [frequencies, n_pix]
            data without from which the CMB (sample) was substracted

        Returns
        -------
        array[float] of dimensions [n_pix]
            computation of spectral likelihood
        """

        # Building the spectral_likelihood: - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

        chx.assert_shape(complete_mixing_matrix, (self.n_frequencies, self.n_components, self.n_pix))
        chx.assert_shape(full_data_without_CMB, (self.n_frequencies, self.nstokes, self.n_pix))
        chx.assert_shape(self.freq_inverse_noise, (self.n_frequencies, self.n_frequencies, self.n_pix))

        ## Getting B_fg, foreground part of the mixing matrix
        complete_mixing_matrix_fg = complete_mixing_matrix[:, 1:, :]

        # Computing (B_f^t N^{-1} B_f)^{-1}
        invBtinvNB_fg = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)
        # Computing B_f^t N^{-1}
        BtinvN_fg = get_BtinvN(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)

        # Computing B_f^t N^{-1} (d - B_c s_c)
        full_data_without_CMB_with_noise = jnp.einsum('cfp,fsp->csp', BtinvN_fg, full_data_without_CMB)
        chx.assert_shape(full_data_without_CMB_with_noise, (self.n_components - 1, self.nstokes, self.n_pix))

        ## Computation of the spectral likelihood: - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        first_term_complete = jnp.einsum(
            'csp,cmp,msp', full_data_without_CMB_with_noise, invBtinvNB_fg, full_data_without_CMB_with_noise
        )
        return -(-first_term_complete + 0) / 2.0

    def get_conditional_proba_spectral_likelihood_JAX_pixel(self, complete_mixing_matrix, full_data_without_CMB):
        """
        Get conditional probability of spectral likelihood from the full mixing matrix

        The associated conditional probability is given by:
        - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

        with d = full_data_without_CMB, B_c = complete_mixing_matrix, B_f = complete_mixing_matrix[:,1:,:]
        d is assumed to be band-limited

        Parameters
        ----------
        complete_mixing_matrix: array[float] of dimensions [component, frequencies]
            complete mixing matrix
        full_data_without_CMB: array[float] of dimensions [frequencies, n_pix]
            data without from which the CMB (sample) was substracted

        Returns
        -------
        array[float] of dimensions [n_pix]
            computation of spectral likelihood per pixel
        """

        # Building the spectral_likelihood: - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)

        chx.assert_shape(complete_mixing_matrix, (self.n_frequencies, self.n_components, self.n_pix))
        chx.assert_shape(full_data_without_CMB, (self.n_frequencies, self.nstokes, self.n_pix))
        chx.assert_shape(self.freq_inverse_noise, (self.n_frequencies, self.n_frequencies, self.n_pix))

        ## Getting B_fg, foreground part of the mixing matrix
        complete_mixing_matrix_fg = complete_mixing_matrix[:, 1:, :]

        # Computing (B_f^t N^{-1} B_f)^{-1}
        invBtinvNB_fg = get_inv_BtinvNB(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)
        # Computing B_f^t N^{-1}
        BtinvN_fg = get_BtinvN(self.freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)

        # Computing B_f^t N^{-1} (d - B_c s_c)
        full_data_without_CMB_with_noise = jnp.einsum('cfp,fsp->csp', BtinvN_fg, full_data_without_CMB)
        chx.assert_shape(full_data_without_CMB_with_noise, (self.n_components - 1, self.nstokes, self.n_pix))

        ## Computation of the spectral likelihood: - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        first_term_complete = jnp.einsum(
            'csp,cmp,msp->p', full_data_without_CMB_with_noise, invBtinvNB_fg, full_data_without_CMB_with_noise
        )
        return -(-first_term_complete + 0) / 2.0

    def get_conditional_proba_correction_likelihood_JAX_v2d(
        self,
        noise_CMB_component_,
        component_eta_maps,
        red_cov_approx_matrix_sqrt,
        first_guess=None,
        return_inverse=False,
        precond_func=None,
        full_sky_correction=False,
    ):
        """
        Get conditional probability of correction term in the likelihood from the full mixing matrix

        Noting C_approx = \tilde{C}, the associated conditional probability is given by:
            - (eta ^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} eta
        Or, with the developped version:
            - (eta^t ( Id + C_approx^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C_approx^{1/2}) \\eta

        Parameters
        ----------
        noise_CMB_component_: array[float] of dimensions [n_pix]
            pixel covariance matrix (B^t N^{-1} B)^{-1}
        component_eta_maps: array[float] of dimensions [nstokes, n_pix]
            set of eta maps
        red_cov_approx_matrix_sqrt: array[float] of dimensions [component, component, n_pix]
            square root of the approximate covariance matrix
        first_guess: array[float] of dimensions [component, n_pix] (optional)
            initial guess for the CG algorithm
        return_inverse: bool (optional)
            boolean to return the inverse term or not (default is False)
        precond_func: function or None (optional)
            preconditioning function to be used in the CG algorithm

        Returns
        -------
        computation of log-proba correction term to the likelihood
        """

        # Building the correction term to the likelihood: - (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} \eta

        ## Preparing the mixing matrix and C_approx^{-1/2}
        N_c = (
            noise_CMB_component_ * jhp.nside2resol(self.nside) ** 2
        )  # Rescaling the pixel covariance matrix into a weighting

        ## Preparing the operator ( C_approx^{-1} + N_c^{-1} )^{-1}
        N_c_repeat = jnp.broadcast_to(
            N_c, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat N_c for each Stokes parameter, for speed-up afterwards

        N_c_inv = jnp.zeros_like(N_c)
        N_c_inv = N_c_inv.at[..., self.mask != 0].set(1 / N_c[self.mask != 0])
        N_c_inv_repeat = jnp.broadcast_to(
            N_c_inv, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Preparing the operator C_approx^{1/2}
        first_part_left = lambda x: maps_x_red_covariance_cell_JAX(
            x.reshape((self.nstokes, self.n_pix)),
            red_cov_approx_matrix_sqrt,
            nside=self.nside,
            lmin=self.lmin,
            n_iter=self.n_iter,
        ).ravel()

        ## Preparing the operator N_c^{-1}
        def second_part_left(x):
            return x * N_c_inv_repeat

        ## Preparing the CG function (Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2})
        func_left_term = lambda x: x.ravel() + first_part_left(second_part_left(first_part_left(x))).ravel()

        ## Preparation of the initial guess
        initial_guess = jnp.copy(component_eta_maps)
        if first_guess is not None:
            initial_guess = jnp.copy(first_guess)

        ## Preparation of the right member of the CG
        right_member = component_eta_maps

        time_start = time.time()
        inverse_term, iterations = jsp.sparse.linalg.cg(
            func_left_term,
            right_member.ravel(),
            x0=initial_guess.ravel(),
            tol=self.tolerance_CG,
            maxiter=self.limit_iter_cg_eta,
            M=precond_func,
        )
        ## Computing (Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2})^{-1} \eta

        print(
            'CG-Inverse term eta finished in ',
            time.time() - time_start,
            ' seconds with ',
            iterations,
            ' iterations',
            flush=True,
        )

        central_term = self.mask
        if full_sky_correction:
            central_term = jnp.ones_like(self.mask)

        ## Computing the log-proba of the correction term
        second_term_complete = jnp.einsum(
            'sp,p,sp', component_eta_maps, central_term, inverse_term.reshape(self.nstokes, self.n_pix)
        )
        if return_inverse:
            return -(-0 + second_term_complete) / 2.0 * jhp.nside2resol(self.nside) ** 2, inverse_term.reshape(
                self.nstokes, self.n_pix
            )
        return (
            -(-0 + second_term_complete) / 2.0 * jhp.nside2resol(self.nside) ** 2
        )  # Multiplying by the pixel area as it was removed formerly

    def get_conditional_proba_correction_likelihood_JAX_v2db(
        self,
        old_params_mixing_matrix,
        new_params_mixing_matrix,
        inverse_term,
        component_eta_maps,
        red_cov_approx_matrix_sqrt,
        inverse_term_x_Capprox_root=None,
        full_sky_correction=True,
    ):
        """Get conditional probability of correction term in the likelihood from the full mixing matrix,
        assuming the difference between the old and new mixing matrix is small

        With notation C_approx instead of \tilde{C}, the associated conditional probability is given by:
            - (\\eta ^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} \\eta
        Or:
            - (\\eta^t ( Id + C_approx^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C_approx^{1/2}) \\eta

        The computation proceeds as follows:
            Knowing A^{-1} = ( Id + C_approx^{1/2} N_{c,old}^{-1} C_approx^{1/2} )^{-1} ;
            We compute ( Id + C_approx^{1/2} N_{c,new}^{-1} C_approx^{1/2} )^{-1} eta
            From:  \\eta (A + C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2})^{-1} \\eta
                    ~=  \\eta^t (A^{-1} - A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1}) \\eta
            Which holds if || N_{c,new}^{-1} - N_{c,old}^{-1} || is small enough

        As we have:
            - inverse_term = A^{-1} eta
            - inverse_term_x_Capprox_root = C_approx^{1/2} A^{-1} eta (optional, otherwise it can be computed in the routine)
            - The B_{old} and B_{new} mixing matrices to reconstruct easily N_{c,old}^{-1} and N_{c,new}^{-1}
        We can thus easily compute:
            \\eta (A^{-1} - A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1}) \\eta

        Parameters
        ----------
        old_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            old mixing matrix B_{old} to generate N_{c,old}
        new_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            new mixing matrix B_{new} to generate N_{c,new}
        inverse_term: array[float] of dimensions [component, n_pix]
            previous inverse term computed with N_{c,old}
        component_eta_maps: array[float] of dimensions [nstokes, n_pix]
            set of eta maps
        red_cov_approx_matrix: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            matrix square root of the covariance of the covariance matrice approx (C_approx) in harmonic domain
        inverse_term_x_Capprox_root: array[float] of dimensions [component, n_pix] (optional)
            precomputed product C_approx^{1/2} A^{-1} eta, if not given it will be recomputed here

        Returns
        -------
        float
            computation of the log-proba of the correction term to the likelihood
        """

        ## Retrieving the mixing matrix B_{old} from the old set of parameters
        # self.update_params(old_params_mixing_matrix, jax_use=True)
        # old_mixing_matrix = self.get_B(jax_use=True)
        old_mixing_matrix = self.get_B_from_params(old_params_mixing_matrix, jax_use=True)

        ## Retrieving the mixing matrix B_{new} from the new set of parameters
        # self.update_params(new_params_mixing_matrix, jax_use=True)
        # new_mixing_matrix = self.get_B(jax_use=True)
        new_mixing_matrix = self.get_B_from_params(new_params_mixing_matrix, jax_use=True)

        ## Preparing both old and new mixing matrices
        old_invBtinvNB = (
            get_inv_BtinvNB(self.freq_inverse_noise, old_mixing_matrix, jax_use=True) * jhp.nside2resol(self.nside) ** 2
        )
        new_invBtinvNB = (
            get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True) * jhp.nside2resol(self.nside) ** 2
        )

        ## Preparing N_c^{-1} for both old and new mixing matrices
        old_N_c_inv = jnp.zeros_like(old_invBtinvNB[0, 0])
        old_N_c_inv = old_N_c_inv.at[..., self.mask != 0].set(
            1 / old_invBtinvNB[0, 0, self.mask != 0]
        )  # Define N_c^{-1} for the old mixing matrix
        old_N_c_inv_repeat = jnp.broadcast_to(
            old_N_c_inv, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat old_N_c_inv for each Stokes parameter, for speed-up afterwards

        new_N_c_inv = jnp.zeros_like(new_invBtinvNB[0, 0])
        new_N_c_inv = new_N_c_inv.at[..., self.mask != 0].set(
            1 / new_invBtinvNB[0, 0, self.mask != 0]
        )  # Define N_c^{-1} for the new mixing matrix
        new_N_c_inv_repeat = jnp.broadcast_to(
            new_N_c_inv, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat new_N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Preparing the operator (N_{c,new}^{-1} - N_{c,old}^{-1})
        def second_part_left(x):
            return x * (new_N_c_inv_repeat - old_N_c_inv_repeat)

        func_to_apply = lambda x: second_part_left(x)

        ## If not precomputed, we compute C_approx^{1/2} A^{-1} eta ; we recommend to pre-compute it for speed-up
        if inverse_term_x_Capprox_root is None:
            inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(
                inverse_term.reshape(self.nstokes, self.n_pix),
                red_cov_approx_matrix_sqrt,
                nside=self.nside,
                lmin=self.lmin,
                n_iter=self.n_iter,
            ).ravel()

        ## Applying the operator (N_{c,new}^{-1} - N_{c,old}^{-1}) to C_approx^{1/2} A^{-1} eta
        perturbation_term = func_to_apply(inverse_term_x_Capprox_root).reshape(self.nstokes, self.n_pix)

        central_term = self.mask
        if full_sky_correction:
            central_term = jnp.ones_like(self.mask)

        ## Computing contribution of \eta A^{-1} \eta
        first_order_term = jnp.einsum(
            'sp,p,sp', component_eta_maps, central_term, inverse_term.reshape(self.nstokes, self.n_pix)
        )

        ## Computing contribution of \eta A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1} \eta
        perturbation_term = jnp.einsum(
            'sp,p,sp', perturbation_term, central_term, inverse_term_x_Capprox_root.reshape(self.nstokes, self.n_pix)
        )

        ## Assembling everything
        new_log_proba = first_order_term - perturbation_term
        # print("First order:", first_order_term)
        # print("Perturbation:", -perturbation_term)

        return (
            -(-0 + new_log_proba) / 2.0 * jhp.nside2resol(self.nside) ** 2
        )  # Multiplying by the pixel area as it was removed formerly

    def get_conditional_proba_correction_likelihood_JAX_pixel(
        self,
        old_params_mixing_matrix,
        new_params_mixing_matrix,
        inverse_term,
        component_eta_maps,
        red_cov_approx_matrix_sqrt,
        inverse_term_x_Capprox_root=None,
        full_sky_correction=True,
    ):
        """
        Get conditional probability of correction term in the likelihood from the full mixing matrix,
        assuming the difference between the old and new mixing matrix is small

        With notation C_approx instead of \tilde{C}, the associated conditional probability is given by:
            - (\\eta ^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} \\eta
        Or:
            - (\\eta^t ( Id + C_approx^{1/2} (E^t (B^t N^{-1} B)^{-1} E)^{-1} C_approx^{1/2}) \\eta

        The computation proceeds as follows:
            Knowing A^{-1} = ( Id + C_approx^{1/2} N_{c,old}^{-1} C_approx^{1/2} )^{-1} ;
            We compute ( Id + C_approx^{1/2} N_{c,new}^{-1} C_approx^{1/2} )^{-1} eta
            From:  \\eta (A + C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2})^{-1} \\eta
                    ~=  \\eta^t (A^{-1} - A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1}) \\eta
            Which holds if || N_{c,new}^{-1} - N_{c,old}^{-1} || is small enough

        As we have:
            - inverse_term = A^{-1} eta
            - inverse_term_x_Capprox_root = C_approx^{1/2} A^{-1} eta (optional, otherwise it can be computed in the routine)
            - The B_{old} and B_{new} mixing matrices to reconstruct easily N_{c,old}^{-1} and N_{c,new}^{-1}
        We can thus easily compute:
            \\eta (A^{-1} - A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1}) \\eta

        Parameters
        ----------
        old_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            old mixing matrix B_{old} to generate N_{c,old}
        new_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            new mixing matrix B_{new} to generate N_{c,new}
        inverse_term: array[float] of dimensions [component, n_pix]
            previous inverse term computed with N_{c,old}
        component_eta_maps: array[float] of dimensions [nstokes, n_pix]
            set of eta maps
        red_cov_approx_matrix: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            matrix square root of the covariance of the covariance matrice approx (C_approx) in harmonic domain
        inverse_term_x_Capprox_root: array[float] of dimensions [component, n_pix] (optional)
            precomputed product C_approx^{1/2} A^{-1} eta, if not given it will be recomputed here

        Returns
        -------
        float
            computation of the log-proba of the correction term to the likelihood
        """

        ## Retrieving the mixing matrix B_{old} from the old set of parameters
        # self.update_params(old_params_mixing_matrix, jax_use=True)
        # old_mixing_matrix = self.get_B(jax_use=True)
        old_mixing_matrix = self.get_B_from_params(old_params_mixing_matrix, jax_use=True)

        ## Retrieving the mixing matrix B_{new} from the new set of parameters
        # self.update_params(new_params_mixing_matrix, jax_use=True)
        # new_mixing_matrix = self.get_B(jax_use=True)
        new_mixing_matrix = self.get_B_from_params(new_params_mixing_matrix, jax_use=True)

        ## Preparing both old and new mixing matrices
        old_invBtinvNB = (
            get_inv_BtinvNB(self.freq_inverse_noise, old_mixing_matrix, jax_use=True) * jhp.nside2resol(self.nside) ** 2
        )
        new_invBtinvNB = (
            get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True) * jhp.nside2resol(self.nside) ** 2
        )

        ## Preparing N_c^{-1} for both old and new mixing matrices
        old_N_c_inv = jnp.zeros_like(old_invBtinvNB[0, 0])
        old_N_c_inv = old_N_c_inv.at[..., self.mask != 0].set(
            1 / old_invBtinvNB[0, 0, self.mask != 0]
        )  # Define N_c^{-1} for the old mixing matrix
        old_N_c_inv_repeat = jnp.broadcast_to(
            old_N_c_inv, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat old_N_c_inv for each Stokes parameter, for speed-up afterwards

        new_N_c_inv = jnp.zeros_like(new_invBtinvNB[0, 0])
        new_N_c_inv = new_N_c_inv.at[..., self.mask != 0].set(
            1 / new_invBtinvNB[0, 0, self.mask != 0]
        )  # Define N_c^{-1} for the new mixing matrix
        new_N_c_inv_repeat = jnp.broadcast_to(
            new_N_c_inv, (self.nstokes, self.n_pix)
        ).ravel()  ## Repeat new_N_c_inv for each Stokes parameter, for speed-up afterwards

        ## Preparing the operator (N_{c,new}^{-1} - N_{c,old}^{-1})
        def second_part_left(x):
            return x * (new_N_c_inv_repeat - old_N_c_inv_repeat)

        func_to_apply = lambda x: second_part_left(x)

        ## If not precomputed, we compute C_approx^{1/2} A^{-1} eta ; we recommend to pre-compute it for speed-up
        if inverse_term_x_Capprox_root is None:
            inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(
                inverse_term.reshape(self.nstokes, self.n_pix),
                red_cov_approx_matrix_sqrt,
                nside=self.nside,
                lmin=self.lmin,
                n_iter=self.n_iter,
            ).ravel()

        ## Applying the operator (N_{c,new}^{-1} - N_{c,old}^{-1}) to C_approx^{1/2} A^{-1} eta
        perturbation_term = func_to_apply(inverse_term_x_Capprox_root).reshape(self.nstokes, self.n_pix)

        central_term = self.mask
        if full_sky_correction:
            central_term = jnp.ones_like(self.mask)
        ## Computing contribution of \eta A^{-1} \eta
        first_order_term = jnp.einsum(
            'sp,p,sp->p', component_eta_maps, central_term, inverse_term.reshape(self.nstokes, self.n_pix)
        )

        ## Computing contribution of \eta A^{-1} C_approx^{1/2} (N_{c,new}^{-1} - N_{c,old}^{-1}) C_approx^{1/2} A^{-1} \eta
        perturbation_term = jnp.einsum(
            'sp,p,sp->p', perturbation_term, central_term, inverse_term_x_Capprox_root.reshape(self.nstokes, self.n_pix)
        )

        ## Assembling everything
        new_log_proba = first_order_term - perturbation_term
        # print("First order:", first_order_term)
        # print("Perturbation:", -perturbation_term)

        return (
            -(-0 + new_log_proba) / 2.0 * jhp.nside2resol(self.nside) ** 2
        )  # Multiplying by the pixel area as it was removed formerly

    def get_conditional_proba_mixing_matrix_v2b_JAX(
        self,
        new_params_mixing_matrix,
        full_data_without_CMB,
        red_cov_approx_matrix_sqrt,
        component_eta_maps=None,
        first_guess=None,
        biased_bool=False,
        precond_func=None,
        full_sky_correction=True,
    ):
        """Get conditional probability of the conditional probability associated with the B_f parameters

        With notation C_approx instead of \tilde{C}, the associated conditional probability is given by:
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + \\eta^t ( Id + C_approx^{1/2} N_c^{-1} C_approx^{1/2} )^{-1} \\eta

        Parameters
        ----------
        new_params_mixing_matrix: array[float] of dimensions [nfreq-len(pos_special_frequencies), ncomp-1]
            new B_f parameters of the mixing matrix to compute the log-proba
        full_data_without_CMB: array[float] of dimensions [frequencies, n_pix]
            data without from which the CMB (sample) was substracted, of dimension
        red_cov_approx_matrix: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            matrix square root of the covariance of the covariance matrice approx (C_approx) in harmonic domain
        component_eta_maps: array[float] of dimensions [nstokes, n_pix]
            set of eta maps
        first_guess: array[float] of dimensions [component, n_pix] (optional)
            initial guess for the CG, default is None to use maps of 0
        biased_bool: bool (optional)
            indicate if the log-proba is biased, so computed without the correction, or not
        precond_func: function or None (optional)
            preconditioning function to be used in the CG algorithm

        Returns
        -------
        log_proba: float
            computation of the conditional probability of the mixing matrix
        inverse_term: array[float] of dimensions [component, n_pix]
            inverse term computed with N_{c,new} to be used in the CG algorithm
        """

        ## Updating parameters of the mixing matrix
        # self.update_params(new_params_mixing_matrix,jax_use=True)
        # new_mixing_matrix = self.get_B(jax_use=True) # Retrieving the new mixing matrix
        new_mixing_matrix = self.get_B_from_params(new_params_mixing_matrix, jax_use=True)

        # Compute spectral likelihood: (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(
            new_mixing_matrix, full_data_without_CMB
        )

        if biased_bool:
            # Computation chosen to be without the correction term
            log_proba_perturbation_likelihood = 0
            inverse_term = 0
        else:
            # Compute correction term to the likelihood: (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
            noise_CMB_component = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)[
                0, 0, ...
            ]  # Preparing N_c
            log_proba_perturbation_likelihood, inverse_term = self.get_conditional_proba_correction_likelihood_JAX_v2d(
                noise_CMB_component,
                component_eta_maps,
                red_cov_approx_matrix_sqrt,
                first_guess=first_guess,
                return_inverse=True,
                precond_func=precond_func,
                full_sky_correction=full_sky_correction,
            )

        return (log_proba_spectral_likelihood + log_proba_perturbation_likelihood), inverse_term

    def get_conditional_proba_mixing_matrix_v3_JAX(
        self,
        new_params_mixing_matrix,
        old_params_mixing_matrix,
        full_data_without_CMB,
        red_cov_approx_matrix_sqrt=None,
        component_eta_maps=None,
        first_guess=None,
        previous_inverse_x_Capprox_root=None,
        biased_bool=False,
        full_sky_correction=False,
    ):
        """Get conditional probability of the conditional probability associated with the B_f parameters

        Note that the difference between the old and new mixing matrix is assumed to be small

        With notation C_approx instead of \tilde{C}, the associated conditional probability is given by:
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta

        Parameters
        ----------
        old_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            old mixing matrix B_{old} to generate N_{c,old}
        new_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            new mixing matrix B_{new} to generate N_{c,new}
        full_data_without_CMB: array[float] of dimensions [frequencies, n_pix]
            data without from which the CMB (sample) was substracted, of dimension
        red_cov_approx_matrix: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            matrix square root of the covariance of the covariance matrice approx (C_approx) in harmonic domain
        component_eta_maps: array[float] of dimensions [nstokes, n_pix]
            set of eta maps
        first_guess: array[float] of dimensions [component, n_pix] (optional)
            previous inverse term computed with N_{c,old}, default is None to use maps of 0
        previous_inverse_x_Capprox_root: array[float] of dimensions [component, n_pix] (optional)
            C_approx^{1/2} A^{-1} eta, default None to have it recomputed
        biased_bool: bool (optional)
            indicate if the log-proba is biased, so computed without the correction, or not

        Returns
        -------
        float
            computation of the conditional probability of the mixing matrix
        """

        ## Updating parameters of the mixing matrix
        # self.update_params(new_params_mixing_matrix,jax_use=True)
        # new_mixing_matrix = self.get_B(jax_use=True)
        new_mixing_matrix = self.get_B_from_params(new_params_mixing_matrix, jax_use=True)

        # Compute spectral likelihood: (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX(
            new_mixing_matrix, jnp.array(full_data_without_CMB)
        )

        if biased_bool:
            # Computation chosen to be without the correction term
            log_proba_perturbation_likelihood = 0
        else:
            # Compute correction term to the likelihood: (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
            log_proba_perturbation_likelihood = self.get_conditional_proba_correction_likelihood_JAX_v2db(
                old_params_mixing_matrix,
                new_params_mixing_matrix,
                first_guess,
                component_eta_maps,
                red_cov_approx_matrix_sqrt,
                inverse_term_x_Capprox_root=previous_inverse_x_Capprox_root,
                full_sky_correction=full_sky_correction,
            )

        return log_proba_spectral_likelihood + log_proba_perturbation_likelihood

    def get_conditional_proba_mixing_matrix_v3_pixel_JAX(
        self,
        new_params_mixing_matrix,
        old_params_mixing_matrix,
        full_data_without_CMB,
        red_cov_approx_matrix_sqrt,
        nside_patch,
        component_eta_maps=None,
        first_guess=None,
        previous_inverse_x_Capprox_root=None,
        biased_bool=False,
        full_sky_correction=False,
    ):
        """Get conditional probability of the conditional probability associated with the B_f parameters

        Note that the difference between the old and new mixing matrix is assumed to be small

        With notation C_approx instead of \tilde{C}, the associated conditional probability is given by:
            - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta

        Parameters
        ----------
        old_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            old mixing matrix B_{old} to generate N_{c,old}
        new_params_mixing_matrix: array[float] of dimensions [component, frequencies]
            new mixing matrix B_{new} to generate N_{c,new}
        full_data_without_CMB: array[float] of dimensions [frequencies, n_pix]
            data without from which the CMB (sample) was substracted, of dimension
        red_cov_approx_matrix: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            matrix square root of the covariance of the covariance matrice approx (C_approx) in harmonic domain
        nside_patch: int
            nside of the parameter patch to retrieve in params
        component_eta_maps: array[float] of dimensions [nstokes, n_pix]
            set of eta maps
        first_guess: array[float] of dimensions [component, n_pix] (optional)
            previous inverse term computed with N_{c,old}, default is None to use maps of 0
        previous_inverse_x_Capprox_root: array[float] of dimensions [component, n_pix] (optional)
            C_approx^{1/2} A^{-1} eta, default None to have it recomputed
        biased_bool: bool (optional)
            indicate if the log-proba is biased, so computed without the correction, or not

        Returns
        -------
        array[float] of dimensions [max_length_patch]
            computation of the conditional probability of the mixing matrix for each patch
        """

        ## Updating parameters of the mixing matrix
        # self.update_params(new_params_mixing_matrix,jax_use=True)
        # new_mixing_matrix = self.get_B(jax_use=True)
        # new_mixing_matrix = self.get_B_from_params(new_params_mixing_matrix, jax_use=True)

        new_mixing_matrix, template = self.get_patch_B_from_params(nside_patch, new_params_mixing_matrix, jax_use=True)

        # Compute spectral likelihood: (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        log_proba_spectral_likelihood = self.get_conditional_proba_spectral_likelihood_JAX_pixel(
            new_mixing_matrix, jnp.array(full_data_without_CMB)
        )

        if biased_bool:
            # Computation chosen to be without the correction term
            log_proba_perturbation_likelihood = 0
        else:
            # Compute correction term to the likelihood: (eta^t C_approx^{-1/2} ( C_approx^{-1} + N_c^{-1} )^{-1} C_approx^{-1/2} eta)
            log_proba_perturbation_likelihood = self.get_conditional_proba_correction_likelihood_JAX_pixel(
                old_params_mixing_matrix,
                new_params_mixing_matrix,
                first_guess,
                component_eta_maps,
                red_cov_approx_matrix_sqrt,
                inverse_term_x_Capprox_root=previous_inverse_x_Capprox_root,
                full_sky_correction=full_sky_correction,
            )

        full_log_proba_pixel = log_proba_spectral_likelihood + log_proba_perturbation_likelihood

        def project_pixel_to_patches(idx_patch):
            mask = jnp.where(template == idx_patch, 1, 0)
            return (full_log_proba_pixel * mask).sum()

        return jax.vmap(project_pixel_to_patches)(jnp.arange(self.max_len_patches_Bf))

    def harmonic_marginal_probability(
        self,
        sample_B_f_r,
        noise_weighted_alm_data,
        theoretical_red_cov_r1_tensor,
        theoretical_red_cov_r0_total,
        red_cov_approx_matrix,
    ):
        """
        Compute marginal probability of the full likelihood over s_c, given by:
            -1/2 (d^t P d - s_{c,ML}^t (C^{-1} + N_c^{-1})^{-1} s_{c,ML} + ln | (C + N_c) (C_approx + N_c)^-1 |)

        With:
            P = N^{-1} - N^{-1} B (B^t N^{-1} B)^{-1} B^t N^{-1}
            In the routine, we don't compute the d^t N^{-1} d part as it stays constant per sample, but only the second part

        Here we denote C_approx = \tilde{C}.

        The data are assumed to be provided in the harmonic domain already noise weighted, and the covariance matrices are assumed to be in the harmonic domain as well.
        All harmonic covariance matrices are assumed to be block diagonal.

        The routine will only work in harmonic domain, so any mixing matrix within sample_B_f_r will be averaged over the pixels.

        The routine doesn't explicitely retrieve C, but assume it to be parametrize by r, with C(r) = r * theoretical_red_cov_r1_tensor + theoretical_red_cov_r0_total

        Parameters
        ----------
        sample_B_f_r: array[float] of dimensions [nfreq-len(pos_special_frequencies)*(ncomp-1) + 1]
            sample of the mixing matrix and r parameter
        noise_weighted_alm_data: array[float] of dimensions [nfreq, nstokes]
            noise weighted alms of the data, do N^{-1} d, given in harmonic domain
        theoretical_red_cov_r1_tensor: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            tensor mode covariance matrices in harmonic domain
        theoretical_red_cov_r0_total: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            scalar mode covariance matrices in harmonic domain
        red_cov_approx_matrix: array[float] of dimensions [lmin:lmax, nstokes, nstokes]
            approximate covariance matrices in harmonic domain

        Returns
        -------
        float
            log-proba computation of the marginal probability of the full likelihood over s_c
        """

        ## Checking the dimensions of the inputs
        chx.assert_axis_dimension(sample_B_f_r, 0, 2 * (self.n_frequencies - jnp.size(self.pos_special_freqs)) + 1)
        chx.assert_axis_dimension(red_cov_approx_matrix, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(theoretical_red_cov_r1_tensor, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(theoretical_red_cov_r0_total, 0, self.lmax + 1 - self.lmin)
        chx.assert_axis_dimension(noise_weighted_alm_data, 0, self.n_frequencies)
        chx.assert_axis_dimension(noise_weighted_alm_data, 1, self.nstokes)

        ## Retrieving the mixing matrix and r parameter
        r_param = sample_B_f_r[-1]
        B_f = sample_B_f_r[:-1]

        ## Updating the mixing matrix
        # self.update_params(B_f, jax_use=True)
        # mixing_matrix_sample = self.get_B(jax_use=True).mean(axis=2) # Getting the mean of the mixing matrix over the pixels
        mixing_matrix_sample = self.get_B_from_params(B_f, jax_use=True).mean(
            axis=2
        )  # Getting the mean of the mixing matrix over the pixels

        ## Reconstructing the CMB covariance matrix parametrized by r_param
        red_CMB_cell = theoretical_red_cov_r0_total + r_param * theoretical_red_cov_r1_tensor

        ## Getting the inverse component noise in harmonic domain
        inv_BtinvNB_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sample)
        effective_noise_CMB_c_ell = inv_BtinvNB_c_ell[0, 0]  # Retrieving N_c
        ## Building the CMB noise covariance matrix in harmonic domain with dimensions [lmax-lmin+1, nstokes, nstokes]
        red_noise_CMB = jnp.einsum('l,sk->lsk', effective_noise_CMB_c_ell, jnp.eye(self.nstokes))

        # Computation of the first term -d^t N^{-1} B (B^t N^{-1} B)^{-1} B^t N^{-1} d

        ## Computation of the central term B (B^t N^{-1} B)^{-1} B with dimensions [nfreq, nfreq, lmax-lmin+1]
        central_term_1_ = jnp.einsum('fc,ckl,gk->fgl', mixing_matrix_sample, inv_BtinvNB_c_ell, mixing_matrix_sample)
        ## Making it an operator with dimensions [frequency, frequency, lmax-lmin+1, nstokes, nstokes]
        central_term_1 = jnp.einsum('fnl,sk->fnlsk', central_term_1_, jnp.eye(self.nstokes))

        ## Applying B (B^t N^{-1} B)^{-1} B to N^{-1} d
        frequency_alm_central_term_1 = frequency_alms_x_obj_red_covariance_cell_JAX(
            noise_weighted_alm_data, central_term_1, lmin=self.lmin
        )

        ## Finally building the full first term: -(d N^{-1})^t B (B^t N^{-1} B)^{-1} B^t N^{-1} d
        first_term_complete = -alm_dot_product_JAX(noise_weighted_alm_data, frequency_alm_central_term_1, self.lmax)

        # Computation of the second term s_{c,ML}^t (C^{-1} + N_c^{-1}) s_{c,ML}

        ## Computation in harmonic domain of s_{c,ML} = E^t (B^t N^{-1} B)^{-1} B^t N^{-1} d
        ## First computation of (B^t N^{-1} B)^{-1} B^t
        multiplicative_term_s_cML_ = jnp.einsum('ckl,fk->cfl', inv_BtinvNB_c_ell, mixing_matrix_sample)
        ## Making it an operator with dimensions [frequency, frequency, lmax-lmin+1, nstokes, nstokes]
        multiplicative_term_s_cML = jnp.einsum('cfl,sk->cflsk', multiplicative_term_s_cML_, jnp.eye(self.nstokes))
        ## Applying (B^t N^{-1} B)^{-1} B^t to N^{-1} d
        s_cML = frequency_alms_x_obj_red_covariance_cell_JAX(
            noise_weighted_alm_data, multiplicative_term_s_cML, lmin=self.lmin
        )[0, ...]

        ## Computation of the central term (C + N_c)^{-1} with dimensions [lmax-lmin+1, nstokes, nstokes]
        central_term_2 = jnp.linalg.pinv(red_noise_CMB + red_CMB_cell)
        ## Applying (C + N_c)^{-1} to s_{c,ML}
        alm_central_term_2 = alms_x_red_covariance_cell_JAX(s_cML, central_term_2, lmin=self.lmin)

        ## Retrieving the full log-proba of the second term
        second_term_complete = alm_dot_product_JAX(s_cML, alm_central_term_2, self.lmax)

        # Computation of the third term ln | (C + N_c) (C_approx + N_c)^-1 |
        ## Building first the operator (C + N_c) (C_approx + N_c)^-1 with dimensions [lmax-lmin+1, nstokes, nstokes]
        red_contribution = jnp.einsum(
            'lsk,lkm->lsm', red_CMB_cell + red_noise_CMB, jnp.linalg.pinv(red_noise_CMB + red_cov_approx_matrix)
        )
        ## Computing the determinant of the operator taking into account the block diagonal structure per ell and m
        third_term = (
            (2 * jnp.arange(self.lmin, self.lmax + 1) + 1) * jnp.log(jnp.abs(jnp.linalg.det(red_contribution)))
        ).sum()

        return -(first_term_complete + second_term_complete + third_term) / 2


def single_Metropolis_Hasting_step(random_PRNGKey, old_sample, step_size, log_proba, **model_kwargs):
    """
    Single Metropolis-Hasting step for a single parameter, with a Gaussian proposal distribution

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    array
        new sample of the parameter
    """
    # Splitting the random key
    rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

    # Generating the proposal sample
    sample_proposal = dist.Normal(jnp.ravel(old_sample, order='F'), step_size).sample(key_proposal)

    # Computing the acceptance probability
    accept_prob = -(
        log_proba(jnp.ravel(old_sample, order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)
    )

    # Accepting or rejecting the proposal
    new_sample = jnp.where(
        jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample, order='F')
    )

    return new_sample.reshape(old_sample.shape, order='F')


def bounded_single_Metropolis_Hasting_step(random_PRNGKey, old_sample, step_size, log_proba, min_value, **model_kwargs):
    """
    Single bounded Metropolis-Hasting step for a single parameter, with a Gaussian proposal distribution
    If the proposal is lower than the minimum value, the proposal is rejected

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    array
        new sample of the parameter
    """
    # Splitting the random key
    rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

    # Generating the proposal sample
    sample_proposal = dist.Normal(jnp.ravel(old_sample, order='F'), step_size).sample(key_proposal)

    # Checking if the proposal is lower than the minimum value
    sample_proposal = jnp.where(sample_proposal < min_value, jnp.ravel(old_sample, order='F'), sample_proposal)

    # Computing the acceptance probability
    accept_prob = -(
        log_proba(jnp.ravel(old_sample, order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)
    )

    # Accepting or rejecting the proposal
    new_sample = jnp.where(
        jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample, order='F')
    )

    return new_sample.reshape(old_sample.shape, order='F')


def multivariate_Metropolis_Hasting_step(random_PRNGKey, old_sample, covariance_matrix, log_proba, **model_kwargs):
    """
    Metropolis-Hasting step for a multivariate parameter, with a multivariate Gaussian proposal distribution

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    covariance_matrix: array[float]
        covariance matrix for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    array
        new sample of the parameter
    """
    # Splitting the random key
    rng_key, key_proposal, key_accept = random.split(random_PRNGKey, 3)

    # Generating the proposal sample with a multivariate Gaussian distribution
    sample_proposal = dist.MultivariateNormal(jnp.ravel(old_sample, order='F'), covariance_matrix).sample(key_proposal)

    # Computing the acceptance probability
    accept_prob = -(
        log_proba(jnp.ravel(old_sample, order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)
    )

    # Accepting or rejecting the proposal
    new_sample = jnp.where(
        jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample, order='F')
    )

    return new_sample.reshape(old_sample.shape, order='F')


def multivariate_Metropolis_Hasting_step_numpyro(state, covariance_matrix, log_proba, **model_kwargs):
    """
    Metropolis-Hasting step for a multivariate parameter, with a multivariate Gaussian proposal distribution,
    from a Numpyro state

    Parameters
    ----------
    state: MHState
        Numpyro state
    covariance_matrix: array[float]
        covariance matrix for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    array
        new sample of the parameter
    """
    # Retrieving the old sample and the random key from the Numpyro state
    old_sample, random_PRNGKey = state

    # Splitting the random key
    random_PRNGKey, key_proposal, key_accept = random.split(random_PRNGKey, 3)

    # Generating the proposal sample with a multivariate Gaussian distribution
    sample_proposal = dist.MultivariateNormal(jnp.ravel(old_sample, order='F'), covariance_matrix).sample(key_proposal)

    # Computing the acceptance probability
    accept_prob = -(
        log_proba(jnp.ravel(old_sample, order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)
    )

    # Accepting or rejecting the proposal
    new_sample = jnp.where(
        jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample, order='F')
    )
    return new_sample.reshape(old_sample.shape, order='F'), random_PRNGKey


def multivariate_Metropolis_Hasting_step_numpyro_bounded(state, covariance_matrix, log_proba, boundary, **model_kwargs):
    """
    Metropolis-Hasting step for a multivariate parameter, with a multivariate Gaussian proposal distribution,
    from a Numpyro state

    Parameters
    ----------
    state: MHState
        Numpyro state
    covariance_matrix: array[float]
        covariance matrix for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    model_kwargs: dictionary
        additional arguments for the log-probability function
    boundary: array[float] of the same shape as the parameters
        boundary for the parameters

    Returns
    -------
    array
        new sample of the parameter
    """
    # Retrieving the old sample and the random key from the Numpyro state
    old_sample, random_PRNGKey = state

    # Splitting the random key
    random_PRNGKey, key_proposal, key_accept = random.split(random_PRNGKey, 3)

    # Generating the proposal sample with a multivariate Gaussian distribution
    sample_proposal = dist.MultivariateNormal(jnp.ravel(old_sample, order='F'), covariance_matrix).sample(key_proposal)

    # Computing the acceptance probability
    accept_prob = -(
        log_proba(jnp.ravel(old_sample, order='F'), **model_kwargs) - log_proba(sample_proposal, **model_kwargs)
    )

    # Accepting or rejecting the proposal
    new_sample = jnp.where(
        jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, jnp.ravel(old_sample, order='F')
    )

    new_sample = jnp.where(
        get_bool_array_in_boundary(new_sample.reshape(old_sample.shape, order='F'), boundary).all(),
        new_sample,
        jnp.ravel(old_sample, order='F'),
    )

    return new_sample.reshape(old_sample.shape, order='F'), random_PRNGKey


def separate_single_MH_step_index_v2b(random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, **model_kwargs):
    """
    Perform a set of single Metroplis-Hasting steps on a given set of indexes given by indexes_Bf,
    with a Gaussian proposal distribution. Each index will be sampled conditionned on each other.

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    indexes_Bf: array[int]
        indexes of the parameters to be sampled
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    array
        new sample of the parameter
    """

    def map_func(carry, index_Bf):
        """
        Map function for the JAX scan function to perform the Metroplis-Hasting step on one index,
        conditionned on the other indexes

        Parameters
        ----------
        carry: dictionary
            carry of the JAX scan function ; dictionnary with ['PRNGKey', 'sample', 'log_proba']
        index_Bf: int
            index of the parameter to be updated

        Returns
        -------
        dictionary
            new carry and new sample of the parameter
        array
            new parameter sampled
        """

        # Splitting the random key
        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        # Generating the proposal sample
        sample_proposal = dist.Normal(carry['sample'][index_Bf], step_size[index_Bf]).sample(key_proposal)

        # Few checks
        chx.assert_equal_shape((sample_proposal, carry['sample'][index_Bf]))
        chx.assert_size(sample_proposal, 1)

        # Generating the proposal parameters
        proposal_params = jnp.copy(carry['sample'], order='K')
        proposal_params = proposal_params.at[index_Bf].set(sample_proposal)

        # Computing the log-probability of the proposal
        proposal_log_proba = log_proba(proposal_params, **model_kwargs)

        # Computing the acceptance probability
        accept_prob = -(carry['log_proba'] - proposal_log_proba)

        # Few checks
        chx.assert_size(carry['log_proba'], 1)
        chx.assert_size(proposal_log_proba, 1)

        # Generating the uniform distribution sample
        log_proba_uniform = jnp.log(dist.Uniform().sample(key_accept))

        # Accepting or rejecting the proposal
        new_param = jnp.where(log_proba_uniform < accept_prob, sample_proposal, carry['sample'][index_Bf])
        new_log_proba = jnp.where(log_proba_uniform < accept_prob, proposal_log_proba, carry['log_proba'])

        # Updating the parameter to return
        proposal_params = proposal_params.at[index_Bf].set(new_param)
        new_carry = {'PRNGKey': rng_key, 'sample': proposal_params, 'log_proba': new_log_proba}
        return new_carry, new_param
        # return (rng_key, proposal_params), new_param

    # Few checks
    chx.assert_equal_shape((old_sample, step_size))

    # Initial carry
    initial_carry = {
        'PRNGKey': random_PRNGKey,
        'sample': old_sample,
        'log_proba': log_proba(old_sample, **model_kwargs),
    }

    # Performing the Metroplis-Hasting steps on indexes_Bf, so that each index will be sampled conditionned on each other
    last_carry, new_params = jlax.scan(map_func, initial_carry, indexes_Bf)

    # Returning the new sample
    new_sample = jnp.copy(last_carry['sample'], order='K')

    latest_PRNGKey = last_carry['PRNGKey']
    return latest_PRNGKey, new_sample


def separate_single_MH_step_index_v3(
    random_PRNGKey,
    old_sample,
    step_size,
    log_proba,
    indexes_Bf,
    indexes_patches_Bf,
    size_patches,
    max_len_patches_Bf,
    len_indexes_Bf,
    **model_kwargs,
):
    """
    Perform Metroplis-Hasting step for a given set of indexes given by indexes_patches_Bf

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    indexes_Bf: array[int]
        indexes of the parameters to be sampled
    indexes_patches_Bf: array[int]
        indexes of the first patch of each parameter to be updated
    size_patches: array[int]
        size of all patches
    max_len_patches_Bf: int
        maximum length of the patches
    len_indexes_Bf: int
        maximum index of all possible B_f (not only the free ones)
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    latest_PRNGKey: array[jnp.uint32] of dimensions [2]
        latest random key
    new_sample: array
        new sample of the parameter
    """

    def map_func(carry, counter_i):
        index_Bf = indexes_patches_Bf[counter_i]
        indexes_to_consider = (index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32)) % len_indexes_Bf
        mask_in_indexes_B_f = jnp.where(
            jnp.isin(index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32), indexes_Bf), 1, 0
        )
        mask_indexes_to_consider = jnp.where(
            jnp.arange(max_len_patches_Bf) < size_patches[counter_i], mask_in_indexes_B_f, 0
        )

        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        sample_proposal = dist.Normal(
            carry['sample'][indexes_to_consider], step_size[indexes_to_consider] * mask_indexes_to_consider
        ).sample(key_proposal)

        proposal_params = jnp.copy(carry['sample'])
        proposal_params = proposal_params.at[indexes_to_consider].set(sample_proposal)

        proposal_log_proba = log_proba(proposal_params, **model_kwargs)

        accept_prob = -(carry['log_proba'] - proposal_log_proba)

        log_proba_uniform = jnp.log(dist.Uniform().sample(key_accept))
        new_param = jnp.where(log_proba_uniform < accept_prob, sample_proposal, carry['sample'][indexes_to_consider])
        new_log_proba = jnp.where(log_proba_uniform < accept_prob, proposal_log_proba, carry['log_proba'])

        proposal_params = proposal_params.at[indexes_to_consider].set(new_param)
        new_carry = {'PRNGKey': rng_key, 'sample': proposal_params, 'log_proba': new_log_proba}
        return new_carry, new_param

    initial_carry = {
        'PRNGKey': random_PRNGKey,
        'sample': old_sample,
        'log_proba': log_proba(old_sample, **model_kwargs),
    }

    carry, new_params = jlax.scan(map_func, initial_carry, jnp.arange(indexes_patches_Bf.size))

    new_sample = carry['sample']

    latest_PRNGKey = carry['PRNGKey']

    return latest_PRNGKey, new_sample


def separate_single_MH_step_index_v4_pixel(
    random_PRNGKey,
    old_sample,
    step_size,
    log_proba,
    indexes_Bf,
    indexes_patches_Bf,
    size_patches,
    max_len_patches_Bf,
    len_indexes_Bf,
    **model_kwargs,
):
    """
    Perform Metroplis-Hasting step for a given set of indexes given by indexes_patches_Bf

    Assumes all patches have the same disposition on the sky

    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    indexes_Bf: array[int]
        indexes of the parameters to be sampled
    indexes_patches_Bf: array[int]
        indexes of the first patch of each parameter to be updated
    size_patches: array[int]
        size of all patches
    max_len_patches_Bf: int
        maximum length of the patches
    len_indexes_Bf: int
        maximum index of all possible B_f (not only the free ones)
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    latest_PRNGKey: array[jnp.uint32] of dimensions [2]
        latest random key
    new_sample: array
        new sample of the parameter
    """

    def map_func(carry, counter_i):
        index_Bf = indexes_patches_Bf[counter_i]
        indexes_to_consider = (index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32)) % len_indexes_Bf
        mask_in_indexes_B_f = jnp.where(
            jnp.isin(index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32), indexes_Bf), 1, 0
        )
        mask_indexes_to_consider = jnp.where(
            jnp.arange(max_len_patches_Bf) < size_patches[counter_i], mask_in_indexes_B_f, 0
        )

        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        sample_proposal = dist.Normal(
            carry['sample'][indexes_to_consider], step_size[indexes_to_consider] * mask_indexes_to_consider
        ).sample(key_proposal)

        proposal_params = jnp.copy(carry['sample'])
        proposal_params = proposal_params.at[indexes_to_consider].set(sample_proposal)

        nside_b = jnp.where(size_patches[counter_i] == 1, 1, jnp.sqrt(size_patches[counter_i] / 12))
        proposal_log_proba = log_proba(proposal_params, nside_patch=nside_b, **model_kwargs)

        accept_prob = -(carry['log_proba'] - proposal_log_proba)

        log_proba_uniform = jnp.log(dist.Uniform().sample(key_accept, sample_shape=(max_len_patches_Bf,)))
        new_param = jnp.where(log_proba_uniform < accept_prob, sample_proposal, carry['sample'][indexes_to_consider])
        new_log_proba = jnp.where(log_proba_uniform < accept_prob, proposal_log_proba, carry['log_proba'])

        proposal_params = proposal_params.at[indexes_to_consider].set(new_param)
        new_carry = {'PRNGKey': rng_key, 'sample': proposal_params, 'log_proba': new_log_proba}
        return new_carry, new_param

    nside_init = jnp.where(size_patches[0] == 1, 1, jnp.sqrt(size_patches[0] / 12))
    initial_carry = {
        'PRNGKey': random_PRNGKey,
        'sample': old_sample,
        'log_proba': log_proba(old_sample, nside_patch=nside_init, **model_kwargs),
    }

    carry, new_params = jlax.scan(map_func, initial_carry, jnp.arange(indexes_patches_Bf.size))

    new_sample = carry['sample']

    latest_PRNGKey = carry['PRNGKey']

    return latest_PRNGKey, new_sample


def separate_single_MH_step_index_v4b_pixel(
    random_PRNGKey,
    old_sample,
    step_size,
    log_proba,
    indexes_Bf,
    indexes_patches_Bf,
    size_patches,
    max_len_patches_Bf,
    len_indexes_Bf,
    **model_kwargs,
):
    """
    Perform Metroplis-Hasting step for a given set of indexes given by indexes_patches_Bf

    Assumes all patches have the same disposition on the sky

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    indexes_Bf: array[int]
        indexes of the parameters to be sampled
    indexes_patches_Bf: array[int]
        indexes of the first patch of each parameter to be updated
    size_patches: array[int]
        size of all patches
    max_len_patches_Bf: int
        maximum length of the patches
    len_indexes_Bf: int
        maximum index of all possible B_f (not only the free ones)
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    latest_PRNGKey: array[jnp.uint32] of dimensions [2]
        latest random key
    new_sample: array
        new sample of the parameter

    """

    def map_func(carry, counter_i):
        index_Bf = indexes_patches_Bf[counter_i]
        indexes_to_consider = (index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32)) % len_indexes_Bf
        mask_in_indexes_B_f = jnp.where(
            jnp.isin(index_Bf + jnp.arange(max_len_patches_Bf, dtype=jnp.int32), indexes_Bf), 1, 0
        )
        mask_indexes_to_consider = jnp.where(
            jnp.arange(max_len_patches_Bf) < size_patches[counter_i], mask_in_indexes_B_f, 0
        )

        rng_key, key_proposal, key_accept = random.split(carry['PRNGKey'], 3)

        sample_proposal = dist.Normal(
            carry['sample'][indexes_to_consider], step_size[indexes_to_consider] * mask_indexes_to_consider
        ).sample(key_proposal)

        proposal_params = jnp.copy(carry['sample'])
        proposal_params = proposal_params.at[indexes_to_consider].set(sample_proposal)

        nside_b = jnp.where(size_patches[counter_i] == 1, 1, jnp.sqrt(size_patches[counter_i] / 12))
        proposal_log_proba = log_proba(proposal_params, nside_patch=nside_b, **model_kwargs)

        old_log_proba = jlax.cond(
            carry['size_patch'] == size_patches[counter_i],
            lambda x: carry['log_proba'],
            lambda x: log_proba(x, nside_patch=nside_b, **model_kwargs),
            operand=carry['sample'],
        )

        accept_prob = -(old_log_proba - proposal_log_proba)

        log_proba_uniform = jnp.log(dist.Uniform().sample(key_accept, sample_shape=(max_len_patches_Bf,)))
        new_param = jnp.where(log_proba_uniform < accept_prob, sample_proposal, carry['sample'][indexes_to_consider])
        new_log_proba = jnp.where(log_proba_uniform < accept_prob, proposal_log_proba, old_log_proba)

        proposal_params = proposal_params.at[indexes_to_consider].set(new_param)
        new_carry = {
            'PRNGKey': rng_key,
            'sample': proposal_params,
            'log_proba': new_log_proba,
            'size_patch': size_patches[counter_i],
        }
        return new_carry, new_param

    nside_init = jnp.where(size_patches[0] == 1, 1, jnp.sqrt(size_patches[0] / 12))
    initial_carry = {
        'PRNGKey': random_PRNGKey,
        'sample': old_sample,
        'log_proba': log_proba(old_sample, nside_patch=nside_init, **model_kwargs),
        'size_patch': size_patches[0],
    }

    carry, new_params = jlax.scan(map_func, initial_carry, jnp.arange(indexes_patches_Bf.size))

    new_sample = carry['sample']

    latest_PRNGKey = carry['PRNGKey']

    return latest_PRNGKey, new_sample


def separate_single_MH_step_index_accelerated(
    random_PRNGKey, old_sample, step_size, log_proba, indexes_Bf, first_guess, **model_kwargs
):
    """
    Perform Metroplis-Hasting step for a given set of indexes given by indexes_patches_Bf

    Parameters
    ----------
    random_PRNGKey: array[jnp.uint32] of dimensions [2]
        JAX random key to be splitted to generate the proposal and uniform distribution sample
    old_sample: array
        old sample of the parameter
    step_size: array[float]
        standard deviation for the proposal distribution
    log_proba: array[float]
        log-probability function of the model
    indexes_Bf: array[int]
        indexes of the parameters to be sampled
    first_guess: array[float]
        first guess for the inverse term
    model_kwargs: dictionary
        additional arguments for the log-probability function

    Returns
    -------
    latest_PRNGKey: array[jnp.uint32] of dimensions [2]
        latest random key
    new_sample: array
        new sample of the parameter
    new_inverse_term: array[float]
        array to speed-up computations
    """

    def map_func(carry, index_Bf):
        rng_key, key_proposal, key_accept = random.split(carry[0], 3)

        sample_proposal = dist.Normal(carry[1][index_Bf], step_size[index_Bf]).sample(key_proposal)

        old_inverse = carry[2]

        proposal_params = jnp.copy(carry[1])
        proposal_params = proposal_params.at[index_Bf].set(sample_proposal)

        accept_prob_0, inverse_term_0 = log_proba(carry[1], first_guess=old_inverse, **model_kwargs)
        accept_prob_1, inverse_term_1 = log_proba(proposal_params, first_guess=old_inverse, **model_kwargs)
        accept_prob = -(accept_prob_0 - accept_prob_1)
        new_param = jnp.where(
            jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, carry[1][index_Bf]
        )
        new_inverse_term = jnp.where(
            jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, inverse_term_1, inverse_term_0
        )

        proposal_params = proposal_params.at[index_Bf].set(new_param)
        return (rng_key, proposal_params, new_inverse_term), new_param

    carry, new_params = jlax.scan(
        map_func, (random_PRNGKey, jnp.ravel(old_sample, order='F'), first_guess), indexes_params
    )
    new_sample = jnp.copy(jnp.ravel(old_sample, order='F'))
    new_sample = new_sample.at[indexes_params].set(new_params)
    latest_PRNGKey = carry[0]
    new_inverse_term = carry[2]
    # return latest_PRNGKey, new_sample.reshape(old_sample.shape,order='F'), new_inverse_term
    return latest_PRNGKey, new_sample, new_inverse_term
