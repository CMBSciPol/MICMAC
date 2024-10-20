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
from collections import namedtuple
from functools import partial

import chex as chx
import healpy as hp
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import toml
from jax import config

from micmac.external.fgbuster import get_instrument
from micmac.foregrounds.templates import get_nodes_b, tree_spv_config
from micmac.likelihood.sampling import (
    SamplingFunctions,
    multivariate_Metropolis_Hasting_step_numpyro_bounded,
)
from micmac.noise.noisecovar import get_true_Cl_noise
from micmac.toolbox.tools import (
    frequency_alms_x_obj_red_covariance_cell_JAX,
    get_c_ells_from_red_covariance_matrix,
    get_reduced_matrix_from_c_ell,
    get_reduced_matrix_from_c_ell_jax,
)
from micmac.toolbox.utils import generate_power_spectra_CAMB

__all__ = [
    'HarmonicMicmacSampler',
    'create_Harmonic_MICMAC_sampler_from_MICMAC_sampler_obj',
    'create_Harmonic_MICMAC_sampler_from_toml_file',
]

config.update('jax_enable_x64', True)


class HarmonicMicmacSampler(SamplingFunctions):
    def __init__(
        self,
        nside,
        lmax,
        nstokes,
        frequency_array,
        freq_noise_c_ell,
        pos_special_freqs=[0, -1],
        n_components=3,
        lmin=2,
        n_iter=8,
        mask=None,
        spv_nodes_b=None,
        biased_version=False,
        r_true=0,
        boundary_B_f=None,
        boundary_r=None,
        step_size_r=1e-4,
        covariance_B_f=None,
        number_iterations_sampling=100,
        number_iterations_done=0,
        seed=0,
        disable_chex=True,
        instrument_name='SO_SAT',
    ):
        """
        Main MICMAC Harmonic sampling object to initialize and launch the Metropolis-Hastings (MH) sampling in harmonic domain.
        The MH sampling will store B_f and r parameters.

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
        freq_noise_c_ell: array[float] of dimensions [frequencies, frequencies, lmax+1-lmin] or [frequencies, frequencies, lmax] (in which case it will be cut to lmax+1-lmin)
            optional, noise power spectra for each frequency, in uK^2, dimensions
        pos_special_freqs: list[int] (optional)
            indexes of the special frequencies in the frequency array respectively for synchrotron and dust, default is [0,-1] for first and last frequencies
        n_components: int (optional)
            number of components for the mixing matrix, default 3
        lmin: int (optional)
            minimum multipole for the spherical harmonics transforms and harmonic domain objects, default 2
        n_iter: int (optional)
            number of iterations the spherical harmonics transforms (for map2alm transformations), default 8
        mask: None or array[float] of dimensions [n_pix] (optional)
            mask to use in the sampling  ; if not given, no mask is used, default None
            Note: the mask WILL NOT be applied to the input maps, it will be only used for the propagated noise covariance
            WARNING: Masked input are not currently supported, expect E-to-B leakage

        spv_nodes_b: list[dictionaries] (optional)
            tree for the spatial variability, to generate from a yaml file, default []
            in principle set up by get_nodes_b
            WARNING: The spatial variability is not currently supported, but will be passed to MicmacSampler obj when using create_Harmonic_MICMAC_sampler_from_MICMAC_sampler_obj

        biased_version: bool (optional)
            use the biased version of the likelihood, so no computation of the correction term, default False
        r_true: float (optional)
            true value of r (only used to compute input CMB maps, not actually used in the sampling), default 0

        boundary_B_f: None or array[float] (optional)
            minimum and maximum B_f values accepted for B_f sample, set to [-inf,inf] for each B_f parameter if None, default None
        boundary_r: None or array[float] (optional)
            minimum and maximum r values accepted for r sample, set to [-inf,inf] if None, default None

        step_size_r: float (optional)
            step size for the Metropolis-Hastings sampling of r, default 1e-4
        covariance_B_f: None or array[float] of dimensions [(n_frequencies-len(pos_special_freqs))*(n_components-1), (n_frequencies-len(pos_special_freqs))*(n_components-1)] (optional)
            covariance for the Metropolis-Hastings sampling of B_f ; will be repeated if multiresoltion case, default None
        number_iterations_sampling: int (optional)
            maximum number of iterations for the sampling, default 100
        number_iterations_done: int (optional)
            number of iterations already accomplished, in case the chain is resuming from a previous run, usually set by exterior routines, default 0

        seed: int or array[jnp.uint32] (optional)
            seed for the JAX PRNG random number generator to start the chain or array of a previously computed seed, default 0
        disable_chex: bool (optional)
            disable chex tests (to improve speed)

        instrument_name: str (optional)
            name of the instrument as expected by cmbdb or given as 'customized_instrument' if redefined by user, default 'SO_SAT'
            see https://github.com/dpole/cmbdb/blob/master/cmbdb/experiments.yaml
        """

        # Initialising the parent class
        super().__init__(
            nside=nside,
            lmax=lmax,
            nstokes=nstokes,
            lmin=lmin,
            frequency_array=frequency_array,
            pos_special_freqs=pos_special_freqs,
            n_components=n_components,
            freq_inverse_noise=None,
            freq_noise_c_ell=freq_noise_c_ell,
            n_iter=n_iter,
            mask=mask,
            spv_nodes_b=spv_nodes_b,
        )

        # Run settings
        self.biased_version = bool(
            biased_version
        )  # If True, use the biased version of the likelihood, so no computation of the correction term

        # CMB parameters
        self.r_true = float(r_true)
        assert (freq_noise_c_ell.shape == (self.n_frequencies, self.n_frequencies, self.lmax + 1 - self.lmin)) or (
            freq_noise_c_ell.shape == (self.n_frequencies, self.n_frequencies, self.lmax + 1)
        )
        self.freq_noise_c_ell = freq_noise_c_ell

        # Metropolis-Hastings step-size and covariance parameters
        self.covariance_B_f = covariance_B_f
        self.step_size_r = step_size_r
        if boundary_B_f is None:
            boundary_B_f = jnp.zeros((2, (self.n_frequencies - len(self.pos_special_freqs)) * (self.n_components - 1)))
            boundary_B_f = boundary_B_f.at[0, :].set(-jnp.inf)
            boundary_B_f = boundary_B_f.at[1, :].set(jnp.inf)
        if boundary_r is None:
            boundary_r = jnp.array([-jnp.inf, jnp.inf])
        assert np.array(boundary_B_f).shape == (
            2,
            (self.n_frequencies - len(self.pos_special_freqs)) * (self.n_components - 1),
        )
        assert np.array(boundary_r).shape == (2,)
        self.boundary_B_f_r = jnp.hstack((boundary_B_f, jnp.expand_dims(boundary_r, axis=0).T))

        # Sampling parameters
        self.number_iterations_sampling = int(
            number_iterations_sampling
        )  # Maximum number of iterations for the sampling
        self.number_iterations_done = int(
            number_iterations_done
        )  # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed

        # Optional parameters
        self.disable_chex = disable_chex
        self.instrument_name = instrument_name

        # Samples preparation
        self.all_params_mixing_matrix_samples = jnp.empty(0)
        self.all_samples_r = jnp.empty(0)
        self.last_key_used = None

    def generate_CMB(self, return_spectra=True):
        """
        Returns CMB spectra of scalar modes only and tensor modes only (with r=1)
        Both CMB spectra are either returned in the usual form [number_correlations,lmax+1],
        or in the red_cov form if return_spectra == False
        """

        # Selecting the relevant auto- and cross-correlations from CAMB spectra
        if self.nstokes == 2:
            # EE, BB
            partial_indices_polar = np.array([1, 2])
        elif self.nstokes == 1:
            # TT
            partial_indices_polar = np.array([0])
        else:
            # TT, EE, BB, EB
            partial_indices_polar = np.arange(4)

        # Generating the CMB power spectra
        all_spectra_r0 = generate_power_spectra_CAMB(self.nside * 2, r=0, typeless_bool=True)
        all_spectra_r1 = generate_power_spectra_CAMB(self.nside * 2, r=1, typeless_bool=True)

        # Retrieve the scalar mode spectrum
        camb_cls_r0 = all_spectra_r0['total'][: self.lmax + 1, partial_indices_polar]

        # Retrieve the tensor mode spectrum
        tensor_spectra_r1 = all_spectra_r1['tensor'][: self.lmax + 1, partial_indices_polar]

        theoretical_r1_tensor = np.zeros((self.n_correlations, self.lmax + 1))
        theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

        theoretical_r1_tensor[: self.nstokes, ...] = tensor_spectra_r1.T
        theoretical_r0_total[: self.nstokes, ...] = camb_cls_r0.T

        if return_spectra:
            # Return spectra in the form [number_correlations,lmax+1]
            return theoretical_r0_total, theoretical_r1_tensor

        # Return spectra in the form of the reduced covariance matrix, [lmax+1-lmin,number_correlations,number_correlations]
        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[self.lmin :]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)[self.lmin :]
        return theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor

    def generate_input_freq_maps_from_fgs(self, freq_maps_fgs, return_only_freq_maps=True, return_only_maps=False):
        """
        Generate input frequency maps (CMB+foregrounds) from the input frequency foregrounds maps,
        return either the full frequency maps, the full frequency and CMB maps alone,
        or the full frequency and CMB maps with the theoretical reduced covariance matrices for the CMB scalar and tensor modes

        Parameters
        ----------
        freq_maps_fgs: array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency foregrounds maps
        return_only_freq_maps: bool (optional)
            return only the full frequency maps, bool
        return_only_maps: bool (optional)
            return only the full frequency and CMB maps alone, bool

        Returns
        -------
        input_freq_maps: array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency maps
        input_cmb_maps: array[float] of dimensions [nstokes,n_pix]
            input CMB maps
        theoretical_red_cov_r0_total: array[float] of dimensions [lmax+1-lmin,nstokes,nstokes]
            theoretical reduced covariance matrix for the CMB scalar modes
        theoretical_red_cov_r1_tensor: array[float] of dimensions [lmax+1-lmin,nstokes,nstokes]
            theoretical reduced covariance matrix for the CMB tensor modes
        """
        indices_polar = np.array([1, 2, 4])

        # Generate CMB from CAMB
        theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = self.generate_CMB(return_spectra=False)

        # Retrieve fiducial CMB power spectra
        true_cmb_specra = get_c_ells_from_red_covariance_matrix(
            theoretical_red_cov_r0_total + self.r_true * theoretical_red_cov_r1_tensor
        )
        true_cmb_specra_extended = np.zeros((6, self.lmax + 1))
        true_cmb_specra_extended[indices_polar, self.lmin :] = true_cmb_specra

        # Generate input frequency maps
        input_cmb_maps_alt = hp.synfast(true_cmb_specra_extended, nside=self.nside, new=True, lmax=self.lmax)[1:, ...]
        input_cmb_maps = np.broadcast_to(input_cmb_maps_alt, (self.n_frequencies, self.nstokes, self.n_pix))
        input_freq_maps = input_cmb_maps + freq_maps_fgs

        if return_only_freq_maps:
            return input_freq_maps

        if return_only_maps:
            return input_freq_maps, input_cmb_maps

        return input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor

    def update_variable(self, all_samples, new_samples_to_add):
        """
        Update the samples with new samples to add by stacking them

        Parameters
        ----------
        all_samples: array[float] of dimensions [n_samples,n_pix]
            previous samples to update
        new_samples_to_add: array[float] of dimensions [n_samples,n_pix]
            new samples to add

        Returns
        -------
        all_samples: array[float] of dimensions [n_samples+n_samples,n_pix]
            updated samples
        """
        if jnp.size(all_samples) == 0:
            return new_samples_to_add
        elif jnp.size(new_samples_to_add.shape) == 1:
            return jnp.hstack([all_samples, new_samples_to_add])
        else:
            return jnp.vstack([all_samples, new_samples_to_add])

    def update_samples_MH(self, all_samples):
        """
        Update the samples with new samples to add for r and B_f

        Parameters
        ----------
        all_samples: dictionary
            dictionary of all the samples to update
        """
        # Update the samples of r
        self.all_samples_r = self.update_variable(self.all_samples_r, all_samples[..., -1])
        # Update the samples of B_f
        self.all_params_mixing_matrix_samples = self.update_variable(
            self.all_params_mixing_matrix_samples, all_samples[..., :-1]
        )

    def get_alm_from_frequency_maps(self, input_freq_maps):
        """
        Get the alms from the input frequency maps using JAX

        Parameters
        ----------
        input_freq_maps : array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency maps

        Returns
        -------
        freq_alms_input_maps : array[float] of dimensions [n_frequencies,nstokes,(lmax+1)*(lmax+2)//2]
            alms from the input frequency maps
            the (lmax+1)*(lmax+2)//2 dimension is the flattened number of lm coefficients stored according to the Healpy convention
        """

        assert input_freq_maps.shape == (self.n_frequencies, self.nstokes, self.n_pix)

        ## Preparing JAX wrapper for the Healpy map2alm function
        def wrapper_map2alm(maps_, lmax=self.lmax, n_iter=self.n_iter, nside=self.nside):
            alm_T, alm_E, alm_B = hp.map2alm(maps_.reshape((3, 12 * nside**2)), lmax=lmax, iter=n_iter)
            return np.array([alm_T, alm_E, alm_B])

        ## Preparing JAX pure call back for the Healpy map2alm function
        @partial(jax.jit, static_argnums=(1))
        def pure_call_map2alm(maps_, lmax):
            shape_output = (
                3,
                (lmax + 1) * (lmax // 2 + 1),
            )  ## Shape of the output alms : [3 for all Stokes params, (lmax+1)*(lmax+2)//2 for all alms in the Healpy convention]
            return jax.pure_callback(wrapper_map2alm, jax.ShapeDtypeStruct(shape_output, np.complex128), maps_.ravel())

        JAX_input_freq_maps = jnp.array(input_freq_maps)

        def get_freq_alm(num_frequency):
            input_map_extended = jnp.vstack(
                (JAX_input_freq_maps[num_frequency, 0], JAX_input_freq_maps[num_frequency, ...])
            )  ## Adding empty temperature map

            all_alms = jnp.array(
                pure_call_map2alm(input_map_extended, lmax=self.lmax)
            )  ## Getting alms for all stokes parameters

            return all_alms[3 - self.nstokes :, ...]  ## Removing the empty temperature alms

        return jax.vmap(get_freq_alm)(jnp.arange(self.n_frequencies))  ## Getting alms for all frequencies

    def perform_harmonic_minimize(
        self,
        input_freq_maps,
        c_ell_approx,
        init_params_mixing_matrix,
        theoretical_r0_total,
        theoretical_r1_tensor,
        initial_guess_r=0,
        method_used='ScipyMinimize',
        **options_minimizer,
    ):
        """
        Perform a minimization to find the best r and B_f in harmonic domain. The results will be returned as the best parameters found.

        Parameters
        ----------
        input_freq_maps : array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency maps
        c_ell_approx : array[float] of dimensions [number_correlations, lmax+1]
            approximate CMB power spectra for the correction term
        init_params_mixing_matrix : array[float] of dimensions [n_frequencies-len(pos_special_freqs), n_correlations-1]
            initial parameters for the mixing matrix
        theoretical_r0_total : array[float] of dimensions [lmax+1-lmin, number_correlations, number_correlations]
            theoretical covariance matrix for the CMB scalar modes
        theoretical_r1_tensor : array[float] of dimensions [lmax+1-lmin, number_correlations, number_correlations]
            theoretical covariance matrix for the CMB tensor modes
        initial_guess_r : float (optional)
            initial guess for r, default 0
        method_used : str (optional)
            method used for the minimization, default 'ScipyMinimize'
        options_minimizer : dict (optional)
            additional options dictionary for the minimizer

        Returns
        -------
        params : array[float] of dimensions [n_frequencies-len(pos_special_freqs)*(n_correlations-1) + 1]
            best parameters found
        """
        try:
            import jaxopt as jopt
        except ImportError:
            raise ImportError('jaxopt is not installed. Please install it with "pip install jaxopt"')

        # Disabling all chex checks to speed up the code
        if self.disable_chex:
            print('Disabling chex !!!', flush=True)
            chx.disable_asserts()

        ## Getting only the relevant spectra
        if self.nstokes == 2:
            indices_to_consider = np.array([1, 2, 4])
            partial_indices_polar = indices_to_consider[: self.nstokes]
        elif self.nstokes == 1:
            indices_to_consider = np.array([0])
        else:
            indices_to_consider = np.arange(6)  # All auto- and cross-correlations

        ## Testing the shapes of the scalar and tensor modes spectra
        assert len(theoretical_r0_total.shape) == 2
        assert theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin
        assert len(theoretical_r1_tensor.shape) == 2
        assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

        ## If C_approx was given for all correlations, we need to select only the relevant ones for the polarisation
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[indices_to_consider, :]

        ## Testing the initial mixing matrix
        if len(init_params_mixing_matrix.shape) == 1:
            assert len(init_params_mixing_matrix) == (self.n_frequencies - len(self.pos_special_freqs)) * (
                self.n_correlations - 1
            )
        else:
            # assert len(init_params_mixing_matrix.shape) == 2
            assert init_params_mixing_matrix.shape[0] == (self.n_frequencies - len(self.pos_special_freqs))
            assert init_params_mixing_matrix.shape[1] == (self.n_correlations - 1)

        ## Preparing the reduced covariance matrix for C_approx as well as the CMB scalar and tensor modes in the format [lmax+1-lmin,number_correlations,number_correlations]
        red_cov_approx_matrix = get_reduced_matrix_from_c_ell_jax(c_ell_approx)[self.lmin :, ...]
        if self.biased_version:
            red_cov_approx_matrix = jnp.zeros_like(red_cov_approx_matrix)
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)

        ## Getting alms from the input maps
        freq_alms_input_maps = self.get_alm_from_frequency_maps(input_freq_maps)

        # Preparing the noise weighted alms
        ## Operator N^-1 in format [frequencies, frequencies, lmax+1-lmin, nstokes, nstokes]
        freq_red_inverse_noise = jnp.einsum('fgl,sk->fglsk', self.freq_noise_c_ell, jnp.eye(self.nstokes))
        ## Applying N^-1 to the alms of the input data
        noise_weighted_alm_data = frequency_alms_x_obj_red_covariance_cell_JAX(
            freq_alms_input_maps, freq_red_inverse_noise, lmin=self.lmin
        )

        # Setting up the JAXOpt class:
        if method_used in ['BFGS', 'GradientDescent', 'LBFGS', 'NonlinearCG', 'ScipyMinimize']:
            class_solver = getattr(jopt, method_used)
        else:
            raise ValueError('Method used not recognized for minimization')

        # Setting up the function to minimize
        func_to_minimize = lambda sample_B_f_r: -self.harmonic_marginal_probability(
            sample_B_f_r,
            noise_weighted_alm_data=noise_weighted_alm_data,
            red_cov_approx_matrix=red_cov_approx_matrix,
            theoretical_red_cov_r0_total=theoretical_red_cov_r0_total,
            theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor,
        )
        # Setting up the JAX optimizer
        optimizer = class_solver(fun=func_to_minimize, **options_minimizer)

        # Preparing the initial parameters
        init_params_B_f_r = jnp.concatenate(
            (init_params_mixing_matrix.ravel(order='F'), jnp.array(initial_guess_r).reshape(1))
        )

        print('Start of minimization', flush=True)
        params, state = optimizer.run(init_params_B_f_r)
        print('End of minimization', flush=True)

        print('Found parameters', params, flush=True)

        print('With state', state, flush=True)
        return params

    def perform_harmonic_MH(
        self,
        input_freq_maps,
        c_ell_approx,
        init_params_mixing_matrix,
        theoretical_r0_total,
        theoretical_r1_tensor,
        initial_guess_r=0,
        covariance_B_f_r=None,
        print_bool=True,
    ):
        """
        Perform Metropolis Hastings to find the best r and B_f in harmonic domain.
        The chains will be stored as object attributes:
            - all_samples_r for r
            - all_params_mixing_matrix_samples for B_f

        Parameters
        ----------
        input_freq_maps : array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency maps
        c_ell_approx : array[float] of dimensions [number_correlations, lmax+1]
            approximate CMB power spectra for the correction term
        init_params_mixing_matrix : array[float] of dimensions [n_frequencies-len(pos_special_freqs), n_correlations-1]
            initial parameters for the mixing matrix
        theoretical_r0_total : array[float] of dimensions [lmax+1-lmin, number_correlations, number_correlations]
            theoretical covariance matrix for the CMB scalar modes
        theoretical_r1_tensor : array[float] of dimensions [lmax+1-lmin, number_correlations, number_correlations]
            theoretical covariance matrix for the CMB tensor modes
        initial_guess_r : float (optional)
            initial guess for r, default 0
        covariance_B_f_r : None or array[float] of dimensions [(n_frequencies-len(pos_special_freqs))*(n_correlations-1) + 1, (n_frequencies-len(pos_special_freqs))*(n_correlations-1) + 1] (optional)
            covariance for the Metropolis-Hastings sampling of B_f and r, default None
        print_bool: bool (optional)
            option for test prints, default True
        """

        # Disabling all chex checks to speed up the code
        # chx acts like an assert, but is JAX compatible
        if self.disable_chex:
            print('Disabling chex !!!', flush=True)
            chx.disable_asserts()

        ## Getting only the relevant spectra
        if self.nstokes == 2:
            indices_to_consider = np.array([1, 2, 4])
            partial_indices_polar = indices_to_consider[: self.nstokes]
        elif self.nstokes == 1:
            indices_to_consider = np.array([0])
        else:
            indices_to_consider = np.arange(6)  # All auto- and cross-correlations

        ## Testing the shapes of the scalar and tensor modes spectra
        assert len(theoretical_r0_total.shape) == 2
        assert (
            theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin
        )  # or (theoretical_r0_total.shape[1] == self.lmax + 1)
        assert len(theoretical_r1_tensor.shape) == 2
        assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

        ## Getting the theoretical reduced covariance matrix for C_approx as well as the CMB scalar and tensor modes in the format [lmax+1-lmin,number_correlations,number_correlations]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)

        ## Testing shapes of C_approx
        assert len(c_ell_approx.shape) == 2
        ## If C_approx was given for all correlations, we need to select only the relevant ones for the polarisation
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[indices_to_consider, :]

        ## Cutting the C_ell to the relevant ell range
        if c_ell_approx.shape[1] == self.lmax + 1:
            c_ell_approx = c_ell_approx[:, self.lmin :]
        assert c_ell_approx.shape[1] == self.lmax + 1 - self.lmin

        ## Testing the initial mixing matrix
        if len(init_params_mixing_matrix.shape) == 1:
            assert len(init_params_mixing_matrix) == (self.n_frequencies - len(self.pos_special_freqs)) * (
                self.n_correlations - 1
            )
        else:
            # assert len(init_params_mixing_matrix.shape) == 2
            assert init_params_mixing_matrix.shape[0] == (self.n_frequencies - len(self.pos_special_freqs))
            assert init_params_mixing_matrix.shape[1] == (self.n_correlations - 1)

        # Preparing for the full Metropolis-Hatings sampling

        ## Initial guesses preparation
        initial_eta = jnp.zeros((self.nstokes, self.n_pix))
        params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix).ravel(order='F')

        ## CMB covariance preparation in the format [lmax+1-lmin,nstokes,nstokes]
        red_cov_approx_matrix = get_reduced_matrix_from_c_ell_jax(c_ell_approx)
        if self.biased_version:
            red_cov_approx_matrix = jnp.zeros_like(red_cov_approx_matrix)

        ## Preparing the JAX PRNG key from the seed of the object
        PRNGKey = random.PRNGKey(self.seed)

        if self.last_key_used is not None:
            PRNGKey = self.last_key_used

        ## Preparing the step-size for Metropolis-within-Gibbs of B_f sampling
        dimension_param_B_f = (self.n_frequencies - len(self.pos_special_freqs)) * (self.n_correlations - 1)
        if covariance_B_f_r is None:
            if self.covariance_B_f is None:
                raise ValueError('Please provide a covariance_B_f')
            assert (self.covariance_B_f).shape == (dimension_param_B_f, dimension_param_B_f)

            ## Building the full covariance of both B_f and r, without correlations between B_f and r
            covariance_B_f_r = jnp.zeros((dimension_param_B_f + 1, dimension_param_B_f + 1))
            covariance_B_f_r = covariance_B_f_r.at[:dimension_param_B_f, :dimension_param_B_f].set(
                self.covariance_B_f
            )  ## Setting the covariance for B_f
            covariance_B_f_r = covariance_B_f_r.at[dimension_param_B_f, dimension_param_B_f].set(
                self.step_size_r**2
            )  ## Setting the step-size for r
        else:
            assert covariance_B_f_r.shape == (dimension_param_B_f + 1, dimension_param_B_f + 1)

        if print_bool:
            print('Covariance B_f, r:', covariance_B_f_r, flush=True)

        ## Getting alms from the input maps
        input_freq_alms = self.get_alm_from_frequency_maps(input_freq_maps)
        ## Preparing the noise weighted alms
        freq_red_inverse_noise = jnp.einsum(
            'fgl,sk->fglsk', self.freq_noise_c_ell, jnp.eye(self.nstokes)
        )  ## Operator N^-1 in format [frequencies, frequencies, lmax+1-lmin, nstokes, nstokes]
        ## Applying N^-1 to the alms of the input data
        noise_weighted_alm_data = frequency_alms_x_obj_red_covariance_cell_JAX(
            input_freq_alms, freq_red_inverse_noise, lmin=self.lmin
        )

        print(f'Starting {self.number_iterations_sampling} iterations for harmonic run', flush=True)

        MHState = namedtuple('MHState', ['u', 'rng_key'])

        class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
            sample_field = 'u'

            def __init__(self, log_proba, covariance_matrix, boundary_B_f_r=self.boundary_B_f_r):
                self.log_proba = log_proba
                self.covariance_matrix = covariance_matrix
                self.boundary_B_f_r = boundary_B_f_r

            def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
                return MHState(init_params, rng_key)

            def sample(self, state, model_args, model_kwargs):
                """
                One Metropolis-Hastings sampling step
                """
                new_sample, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded(
                    state,
                    covariance_matrix=self.covariance_matrix,
                    log_proba=self.log_proba,
                    boundary=self.boundary_B_f_r,
                    **model_kwargs,
                )
                return MHState(new_sample, rng_key)

        mcmc_obj = numpyro.infer.mcmc.MCMC(
            MetropolisHastings(log_proba=self.harmonic_marginal_probability, covariance_matrix=covariance_B_f_r),
            num_warmup=0,
            num_samples=self.number_iterations_sampling - self.number_iterations_done,
            progress_bar=True,
        )

        # Initializing r and B_f samples
        init_params_mixing_matrix_r = jnp.concatenate(
            (params_mixing_matrix_init_sample, jnp.array(initial_guess_r).reshape(1))
        )

        time_start_sampling = time.time()
        ## Starting the MH sampling !!!
        mcmc_obj.run(
            PRNGKey,
            init_params=init_params_mixing_matrix_r,
            noise_weighted_alm_data=noise_weighted_alm_data,
            theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor,
            theoretical_red_cov_r0_total=theoretical_red_cov_r0_total,
            red_cov_approx_matrix=red_cov_approx_matrix,
        )

        time_full_chain = (time.time() - time_start_sampling) / 60
        print(f'End of MH iterations for harmonic run in {time_full_chain} minutes !', flush=True)

        posterior_samples = mcmc_obj.get_samples()
        if print_bool:
            print('Summary of the run', flush=True)
            mcmc_obj.print_summary()

        # Saving the samples as attributes of the Sampler object
        self.update_samples_MH(posterior_samples)
        self.number_iterations_done = self.number_iterations_sampling
        self.last_key_used = PRNGKey

    def compute_covariance_from_samples(self):
        """
        Compute the covariance matrix from the sample chains of B_f and r

        Returns
        -------
        covariance_B_f_r : array[float]
            covariance matrix of the samples of B_f and r
        """
        if self.number_iterations_done == 0:
            raise ValueError(
                'No iterations done yet, please perform some sampling before computing the covariance matrix'
            )

        print('Computing the covariance matrix from the samples', flush=True)
        all_samples_B_f_r = np.zeros(
            (self.number_iterations_sampling, (self.n_frequencies - len(self.pos_special_freqs)) * 2 + 1)
        )
        all_samples_B_f_r[:, :-1] = self.all_params_mixing_matrix_samples.reshape(
            (self.number_iterations_sampling, (self.n_frequencies - len(self.pos_special_freqs)) * 2)
        )
        all_samples_B_f_r[:, -1] = self.all_samples_r

        return jnp.cov(all_samples_B_f_r, rowvar=False)


def create_Harmonic_MICMAC_sampler_from_toml_file(path_toml_file, path_file_spv):
    """
    Create a HarmonicMicmacSampler object from the path of a toml file and the yaml file for spatial variability

    Parameters
    ----------
    path_toml_file : str
        path to the toml file for the main options of HarmonicMicmacSampler
    path_file_spv : str
        path to the yaml file for the spatial variability options

    Returns
    -------
    Harmonic_MICMAC_Sampler_obj : HarmonicMicmacSampler
        HarmonicMicmacSampler object
    """
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    if dictionary_parameters['instrument_name'] != 'customized_instrument':
        instrument = get_instrument(dictionary_parameters['instrument_name'])
        dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
        dictionary_parameters['freq_noise_c_ell'] = get_true_Cl_noise(
            jnp.array(instrument['depth_p']), dictionary_parameters['lmax']
        )[..., dictionary_parameters['lmin'] :]

    ## Spatial variability (spv) params
    n_fgs_comp = dictionary_parameters['n_components'] - 1
    # total number of params in the mixing matrix for a specific pixel
    n_betas = (
        np.shape(dictionary_parameters['frequency_array'])[0] - len(dictionary_parameters['pos_special_freqs'])
    ) * (n_fgs_comp)
    # Read or create spv config
    root_tree = tree_spv_config(path_file_spv, n_betas, n_fgs_comp, print_tree=True)
    dictionary_parameters['spv_nodes_b'] = get_nodes_b(root_tree)
    return HarmonicMicmacSampler(**dictionary_parameters)


def create_Harmonic_MICMAC_sampler_from_MICMAC_sampler_obj(MICMAC_sampler_obj, depth_p_array, covariance_B_f=None):
    """
    Create a HarmonicMicmacSampler object from a MicmacSampler object
    """

    first_dict = ['nside', 'lmax', 'nstokes', 'frequency_array', 'pos_special_freqs', 'n_components']
    dictionary_parameters = dict()
    for attr in first_dict:
        dictionary_parameters[attr] = getattr(MICMAC_sampler_obj, attr)

    dictionary_parameters['freq_noise_c_ell'] = get_true_Cl_noise(depth_p_array, MICMAC_sampler_obj.lmax)[
        ..., MICMAC_sampler_obj.lmin :
    ]

    # total number of params in the mixing matrix for a specific pixel
    n_free_B_f = (
        np.size(dictionary_parameters['frequency_array']) - len(dictionary_parameters['pos_special_freqs'])
    ) * (dictionary_parameters['n_components'] - 1)
    # Create spv config
    spv_nodes_b = get_nodes_b(
        tree_spv_config('', n_free_B_f, dictionary_parameters['n_components'] - 1, print_tree=False)
    )
    dictionary_parameters['spv_nodes_b'] = spv_nodes_b

    print('Test', dictionary_parameters['freq_noise_c_ell'].shape)
    Harmonic_MICMAC_Sampler_obj = HarmonicMicmacSampler(**dictionary_parameters)

    list_attributes = [
        'pos_special_freqs',
        'n_components',
        'lmin',
        'n_iter',
        'mask',
        'biased_version',
        'r_true',
        'step_size_r',
        'covariance_B_f',
        'instrument_name',
        'number_iterations_sampling',
        'number_iterations_done',
        'seed',
        'disable_chex',
    ]

    for attr in list_attributes:
        Harmonic_MICMAC_Sampler_obj.__setattr__(attr, getattr(MICMAC_sampler_obj, attr))

    if covariance_B_f is not None:
        Harmonic_MICMAC_Sampler_obj.covariance_B_f = covariance_B_f

    return Harmonic_MICMAC_Sampler_obj
