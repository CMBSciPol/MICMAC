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
from functools import partial

import chex as chx
import healpy as hp
import jax
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import numpy as np
from jax import config
from jax_tqdm import scan_tqdm

from micmac.likelihood.sampling import (
    SamplingFunctions,
    separate_single_MH_step_index_accelerated,
    separate_single_MH_step_index_v2b,
    separate_single_MH_step_index_v4_pixel,
    separate_single_MH_step_index_v4b_pixel,
    single_Metropolis_Hasting_step,
)
from micmac.noise.noisecovar import (
    get_BtinvN,
    get_inv_BtinvNB,
    get_inv_BtinvNB_c_ell,
    get_Wd,
)
from micmac.toolbox.statistics import get_1d_recursive_empirical_covariance
from micmac.toolbox.tools import (
    get_c_ells_from_red_covariance_matrix,
    get_cell_from_map_jax,
    get_reduced_matrix_from_c_ell,
    get_reduced_matrix_from_c_ell_jax,
    get_sqrt_reduced_matrix_from_matrix_jax,
    maps_x_red_covariance_cell_JAX,
)
from micmac.toolbox.utils import generate_CMB

__all__ = ['MicmacSampler']

config.update('jax_enable_x64', True)


class MicmacSampler(SamplingFunctions):
    def __init__(
        self,
        nside,
        lmax,
        nstokes,
        frequency_array,
        freq_inverse_noise,
        pos_special_freqs=[0, -1],
        freq_noise_c_ell=None,
        n_components=3,
        lmin=2,
        n_iter=8,
        limit_iter_cg=200,
        limit_iter_cg_eta=200,
        tolerance_CG=1e-8,
        atol_CG=1e-8,
        mask=None,
        save_CMB_chain_maps=False,
        save_eta_chain_maps=False,
        save_all_Bf_params=True,
        save_s_c_spectra=False,
        sample_r_Metropolis=True,
        sample_C_inv_Wishart=False,
        perturbation_eta_covariance=True,
        simultaneous_accept_rate=False,
        non_centered_moves=False,
        save_intermediary_centered_moves=False,
        limit_r_value=False,
        below_0_min_r_value=False,
        min_r_value=None,
        use_alm_sampling_r=False,
        use_alm_sampling_r_wEE=False,
        lmin_BB=None,
        biased_version=False,
        classical_Gibbs=False,
        use_alternative_Bf_sampling=False,
        suppress_low_modes=True,
        use_mask_contribution_eta=False,
        use_binning=False,
        bin_ell_distribution=None,
        acceptance_posdef=False,
        step_size_r=1e-4,
        covariance_Bf=None,
        use_scam_step_size=False,
        burn_in_scam=50,
        s_param_scam=(2.4) ** 2,
        epsilon_param_scam_r=1e-10,
        epsilon_param_scam_Bf=1e-11,
        scam_iteration_updates=50,
        indexes_free_Bf=False,
        number_iterations_sampling=100,
        number_iterations_done=0,
        seed=0,
        disable_chex=True,
        instrument_name='SO_SAT',
        # fwhm=None,
        spv_nodes_b=[],
    ):
        """
        Main MICMAC pixel sampling object to initialize and launch the Gibbs sampling in pixel domain.

        The Gibbs sampling will always store Bf and r (or C) parameters

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
        freq_noise_c_ell: array[float] of dimensions [frequencies, frequencies, lmax+1-lmin] or [frequencies, frequencies, lmax] (in which case it will be cut to lmax+1-lmin)
            optional, noise power spectra for each frequency, in uK^2, dimensions
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
        mask: None or array[float] of dimensions [n_pix] (optional)
            mask to use in the sampling  ; if not given, no mask is used, default None
            Note: the mask WILL NOT be applied to the input maps, it will be only used for the propagated noise covariance

        save_CMB_chain_maps: bool (optional)
            save the CMB chain maps, default False
        save_eta_chain_maps: bool (optional)
            save the eta chain maps, default False

        sample_r_Metropolis: bool (optional)
            sample r with a Metropolis-within-Gibbs step from the BB power spectrum of the
            reconstructed sample of the CMB map during the Gibbs iteration, default True
            Either sample_r_Metropolis or sample_C_inv_Wishart should True but not both
        sample_C_inv_Wishart: bool (optional)
            sample C_inv with Wishart distribution instead of simply r being sampled, default False
            Either sample_r_Metropolis or sample_C_inv_Wishart should True but not both

        limit_r_value: bool (optional)
            limit r value being sampled with the minmum r value given by min_r_value, default False
        min_r_value: float (optional)
            minimum r value accepted for r sample if limit_r_value is True, default 0
        use_alm_sampling_r: bool (optional)
            use the alms sampling for r, default False
        use_alm_sampling_r_wEE: bool (optional)
            use the alms sampling for r with the EE power spectrum, default False

        perturbation_eta_covariance: bool (optional)
            approach to compute difference between CMB noise component for eta log proba instead of repeating the CG for each Bf sampling, default True
        simultaneous_accept_rate: bool (optional)
            use the simultaneous accept rate for the patches of the Bf sampling, default False

        biased_version: bool (optional)
            use the biased version of the likelihood, so no computation of the correction term, default False
        classical_Gibbs: bool (optional)
            sampling only for s_c and the CMB covariance, and neither Bf or eta, default False
        use_alternative_Bf_sampling: bool (optional)
            use the alternative Bf sampling, default False
        suppress_low_modes: bool (optional)
            make eta band-limited

        use_binning: bool (optional)
            use binning for the sampling of inverse Wishart CMB covariance, if False bin_ell_distribution will not be used, default False
        bin_ell_distribution: array[int] (optional)
            binning distribution for the sampling of inverse Wishart CMB covariance, default None
        acceptance_posdef: accept only positive definite matrices C sampling, bool

        step_size_r: float (optional)
            step size for the Metropolis-Hastings sampling of r, default 1e-4
        covariance_Bf: None or array[float] of dimensions [(n_frequencies-len(pos_special_freqs))*(n_components-1), (n_frequencies-len(pos_special_freqs))*(n_components-1)] (optional)
            covariance for the Metropolis-Hastings sampling of Bf ; will be repeated if multiresoltion case, default None
        use_scam_step_size: bool (optional)
            use the SCAM step size for the Metropolis-Hastings sampling of Bf and r (Haario et al. 2005), default False
        burn_in_scam: int (optional)
            number of burn-in iterations before using adaptive step-size (SCAM), not used if use_scam_steps_size is False, default 50
        s_param_scam: float (optional)
            s parameter for the SCAM step size (see Haario et al. 2001, Haario et al. 2005), default (2.4)**2
        epsilon_param_scam_r: float (optional)
            epsilon parameter for the SCAM step size for r (see Haario et al. 2001, Haario et al. 2005), default 1e-10
        epsilon_param_scam_Bf: float (optional)
            epsilon parameter for the SCAM step size for Bf (see Haario et al. 2001, Haario et al. 2005), default 1e-11
        scam_iteration_updates: int (optional)
            number of iterations for which the SCAM step size will be updated (the variance is updated successively for every scam_iteration_updates iterations), default 100

        indexes_free_Bf: bool or array[int] (optional)
            indexes of the free Bf parameters to actually sample and leave the rest of the indices fixed, array of integers, default False to sample all Bf

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
        fwhm: float (optional)
            FWHM of the beam in arcmin, default None (no beam) ; not implemented yet
        spv_nodes_b: list[dictionaries] (optional)
            tree for the spatial variability, to generate from a yaml file, default []
            in principle set up by get_nodes_b
        """

        ## Give the parameters to the parent class
        super().__init__(
            nside=nside,
            lmax=lmax,
            nstokes=nstokes,
            lmin=lmin,
            frequency_array=frequency_array,
            freq_inverse_noise=freq_inverse_noise,
            spv_nodes_b=spv_nodes_b,
            pos_special_freqs=pos_special_freqs,
            n_components=n_components,
            n_iter=n_iter,
            limit_iter_cg=limit_iter_cg,
            limit_iter_cg_eta=limit_iter_cg_eta,
            tolerance_CG=tolerance_CG,
            atol_CG=atol_CG,
            mask=mask,
            bin_ell_distribution=bin_ell_distribution,
        )

        # Run settings
        self.classical_Gibbs = bool(
            classical_Gibbs
        )  # To run the classical Gibbs sampling instead of the full MICMAC sampling
        if self.classical_Gibbs is False:
            # Then we expect to have multiple components
            assert self.n_components > 1
            try:
                assert len(pos_special_freqs) == self.n_components - 1
            except:
                raise Exception('The number of special frequencies should be equal to the number of components - 1')
        self.biased_version = bool(biased_version)  # To have a run without the correction term
        self.perturbation_eta_covariance = bool(
            perturbation_eta_covariance
        )  # To use the perturbation approach for the eta contribution in log-proba of Bf
        self.simultaneous_accept_rate = bool(
            simultaneous_accept_rate
        )  # To use the simultaneous accept rate for the patches of the Bf sampling
        self.use_alternative_Bf_sampling = bool(use_alternative_Bf_sampling)  # To use the alternative Bf sampling
        self.lmin_BB = lmin_BB  # Minimum multipole for the BB power spectrum
        assert self.lmin_BB is None or (self.lmin_BB >= self.lmin and self.lmin_BB <= self.lmax)
        assert ((sample_r_Metropolis and sample_C_inv_Wishart) == False) and (
            (sample_r_Metropolis or not (sample_C_inv_Wishart)) or (not (sample_r_Metropolis) or sample_C_inv_Wishart)
        )
        self.sample_r_Metropolis = bool(sample_r_Metropolis)  # To sample r with Metropolis-Hastings
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)
        self.use_binning = bool(use_binning)  # To use binning for the sampling of inverse Wishart CMB covariance
        self.acceptance_posdef = bool(acceptance_posdef)  # To accept only positive definite matrices for C sampling
        self.non_centered_moves = bool(non_centered_moves)  # To use non-centered moves for C sampling
        self.save_intermediary_centered_moves = bool(
            save_intermediary_centered_moves
        )  # To save intermediary r values in case of non-centered moves in the sampling
        self.limit_r_value = bool(limit_r_value)  # To limit the r value to be positive
        self.min_r_value = min_r_value  # Minimum value for r
        self.below_0_min_r_value = bool(
            below_0_min_r_value
        )  # To allow r to be below 0 while keeping C(r) positive definite
        if limit_r_value:
            assert (
                self.min_r_value is not None or self.below_0_min_r_value
            ), 'If limit_r_value is True, then min_r_value should be given or below_0_min_r_value should be True'
        self.use_alm_sampling_r = bool(use_alm_sampling_r)  # To use the alms sampling for r
        self.use_alm_sampling_r_wEE = bool(
            use_alm_sampling_r_wEE
        )  # To use the alms sampling for r with the EE power spectrum

        self.suppress_low_modes = suppress_low_modes
        self.use_mask_contribution_eta = bool(use_mask_contribution_eta)  # To use the mask in the contribution of eta
        # Harmonic noise parameter
        self.freq_noise_c_ell = freq_noise_c_ell  # Noise power spectra for each frequency, in uK^2, dimensions [frequencies, frequencies, lmax+1-lmin] or [frequencies, frequencies, lmax] (in which case it will be cut to lmax+1-lmin)

        # Metropolis-Hastings parameters
        self.covariance_Bf = covariance_Bf  # Covariance for the Metropolis-Hastings step sampling of Bf
        self.step_size_r = step_size_r  # Step size for the Metropolis-Hastings step sampling of r
        self.use_scam_step_size = bool(
            use_scam_step_size
        )  # Use the SCAM (Single Component Adaptive Metropolis) step size for the Metropolis-Hastings step sampling of Bf and r (Haario et al. 2005)
        self.burn_in_scam = int(burn_in_scam)  # Number of burn-in iterations before using adaptive step-size (SCAM)
        self.s_param_scam = float(s_param_scam)  # s parameter for the SCAM step size
        self.epsilon_param_scam_r = float(epsilon_param_scam_r)  # epsilon parameter for the SCAM step size for r
        self.epsilon_param_scam_Bf = float(epsilon_param_scam_Bf)  # epsilon parameter for the SCAM step size for Bf
        self.scam_iteration_updates = int(scam_iteration_updates)
        # if number_iterations_done > 0:
        #     self.burn_in_scam = 0
        if self.use_scam_step_size:
            print(
                'Using SCAM step size for the Metropolis-Hastings step sampling of Bf and r after',
                self.burn_in_scam,
                'iterations, with parameters s',
                self.s_param_scam,
                'epsilon_r',
                self.epsilon_param_scam_r,
                'epsilon_Bf',
                self.epsilon_param_scam_Bf,
                'and updates every',
                self.scam_iteration_updates,
                'iterations',
                flush=True,
            )

        # Sampling parameters
        if indexes_free_Bf is False:
            # If given as False, then we sample all Bf
            indexes_free_Bf = jnp.arange(self.len_params)
        self.indexes_free_Bf = jnp.array(indexes_free_Bf)
        assert (
            jnp.size(self.indexes_free_Bf) <= self.len_params
        )  # The number of free parameters should be less than the total number of parameters
        assert (
            jnp.max(self.indexes_free_Bf) <= self.len_params
        )  # The indexes should be in the range of the total number of parameters
        assert (
            jnp.min(self.indexes_free_Bf) >= 0
        )  # The indexes should be in the range of the total number of parameters
        self.number_iterations_sampling = int(
            number_iterations_sampling
        )  # Maximum number of iterations for the sampling
        self.number_iterations_done = int(
            number_iterations_done
        )  # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed  # Seed for the JAX PRNG random number generator to start the chain

        # Saving parameters
        self.save_CMB_chain_maps = bool(save_CMB_chain_maps)  # Save the CMB chain maps
        self.save_eta_chain_maps = bool(save_eta_chain_maps)  # Save the eta chain maps
        self.save_all_Bf_params = bool(save_all_Bf_params)  # Save all the Bf chains
        self.save_s_c_spectra = bool(save_s_c_spectra)  # Save the s_c spectra

        # Instrument parameters
        self.instrument_name = instrument_name  # Name of the instrument
        # if fwhm is not None:
        #     self.fwhm = float(fwhm)  # FWHM of the beam in arcmin
        # else:
        #     self.fwhm = None # No beam

        # Check related parameters
        self.disable_chex = disable_chex  # Disable chex tests (to improve speed)

        # Samples preparation
        self.all_samples_eta = jnp.empty(0)
        self.all_params_mixing_matrix_samples = jnp.empty(0)
        self.all_samples_wiener_filter_maps = jnp.empty(0)
        self.all_samples_fluctuation_maps = jnp.empty(0)
        self.all_samples_r = jnp.empty(0)
        self.all_samples_CMB_c_ell = jnp.empty(0)
        self.all_samples_s_c_spectra = jnp.empty(0)

    @property
    def all_samples_s_c(self):
        """
        Returns all the CMB sampled maps from the initial WF and fluctuation maps
        """
        return self.all_samples_wiener_filter_maps + self.all_samples_fluctuation_maps

    def generate_input_freq_maps_from_fgs(
        self, freq_maps_fgs, r_true=0, lmin_input=None, return_only_freq_maps=True, return_only_maps=False
    ):
        """
        Generate input frequency maps (CMB+foregrounds) from the input frequency foregrounds maps,
        return either the full frequency maps, the full frequency and CMB maps alone,
        or the full frequency and CMB maps with the theoretical reduced covariance matrices for the CMB scalar and tensor modes

        Parameters
        ----------
        freq_maps_fgs: array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency foregrounds maps
        r_true: float, optional
            input tensor-to-scalar ratio r to generate the CMB, default to 0
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

        if lmin_input is None:
            lmin_input = self.lmin

        print('Setting lmin to ', lmin_input, flush=True)

        # Define the indices to consider
        indices_polar = np.array([1, 2, 4])

        # Generate CMB from CAMB
        theoretical_r0_total, theoretical_r1_tensor = generate_CMB(
            nside=self.nside, lmax=self.lmax, nstokes=self.nstokes
        )
        # Return spectra in the form of the reduced covariance matrix, [lmax+1,number_correlations,number_correlations]
        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[lmin_input:]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)[lmin_input:]

        # Retrieve fiducial CMB power spectra
        true_cmb_specra = get_c_ells_from_red_covariance_matrix(
            theoretical_red_cov_r0_total + r_true * theoretical_red_cov_r1_tensor
        )
        true_cmb_specra_extended = np.zeros((6, self.lmax + 1))
        true_cmb_specra_extended[indices_polar, lmin_input:] = true_cmb_specra

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

    def update_samples(self, all_samples):
        """
        Update the samples with new samples to add

        Parameters
        ----------
        all_samples: dictionary
            dictionary of all the samples to update
        """
        # Update the eta samples if they were saved and/or if they were sampled
        if self.save_eta_chain_maps and not (self.classical_Gibbs):
            self.all_samples_eta = self.update_variable(self.all_samples_eta, all_samples['eta_maps'])

        # Update the CMB chain maps if they were saved
        if self.save_CMB_chain_maps:
            self.all_samples_wiener_filter_maps = self.update_variable(
                self.all_samples_wiener_filter_maps, all_samples['wiener_filter_term']
            )
            self.all_samples_fluctuation_maps = self.update_variable(
                self.all_samples_fluctuation_maps, all_samples['fluctuation_maps']
            )

        # Update the s_c spectra if they were saved
        if self.save_s_c_spectra:
            self.all_samples_s_c_spectra = self.update_variable(
                self.all_samples_s_c_spectra, all_samples['s_c_spectra']
            )

        # Update the CMB covariance if they were sampled
        if self.sample_C_inv_Wishart:
            if all_samples['red_cov_matrix_sample'].shape[1] == self.lmax + 1 - self.lmin:
                all_samples_CMB_c_ell = jnp.array(
                    [
                        get_c_ells_from_red_covariance_matrix(all_samples['red_cov_matrix_sample'][iteration])
                        for iteration in range(self.number_iterations_sampling)  # - self.number_iterations_done)
                    ]
                )
            else:
                all_samples_CMB_c_ell = all_samples['red_cov_matrix_sample']
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples_CMB_c_ell)
        # Update the r samples if they were sampled
        if self.sample_r_Metropolis:
            if len(all_samples['r_sample'].shape) != len(self.all_samples_r.shape):
                all_samples['r_sample'] = all_samples['r_sample'].squeeze()
            self.all_samples_r = self.update_variable(self.all_samples_r, all_samples['r_sample'])

        # Update the mixing matrix Bf parameters if they were sampled
        if self.save_all_Bf_params:
            self.all_params_mixing_matrix_samples = self.update_variable(
                self.all_params_mixing_matrix_samples, all_samples['params_mixing_matrix_sample']
            )

    def update_one_sample(self, one_sample):
        """
        Update the samples with one sample to add
        """

        if self.save_eta_chain_maps and not (self.classical_Gibbs):
            self.all_samples_eta = self.update_variable(
                self.all_samples_eta, jnp.expand_dims(one_sample['eta_maps'], axis=0)
            )

        if self.save_CMB_chain_maps:
            self.all_samples_wiener_filter_maps = self.update_variable(
                self.all_samples_wiener_filter_maps, jnp.expand_dims(one_sample['wiener_filter_term'], axis=0)
            )
            self.all_samples_fluctuation_maps = self.update_variable(
                self.all_samples_fluctuation_maps, jnp.expand_dims(one_sample['fluctuation_maps'], axis=0)
            )

        if self.sample_C_inv_Wishart:
            if one_sample['red_cov_matrix_sample'].shape[0] == self.lmax + 1 - self.lmin:
                one_sample_CMB_c_ell = get_c_ells_from_red_covariance_matrix(one_sample['red_cov_matrix_sample'])
            else:
                one_sample_CMB_c_ell = one_sample['red_cov_matrix_sample']
            self.all_samples_CMB_c_ell = self.update_variable(
                self.all_samples_CMB_c_ell, jnp.expand_dims(one_sample_CMB_c_ell, axis=0)
            )
        if self.sample_r_Metropolis:
            if self.non_centered_moves:
                if self.save_intermediary_centered_moves:
                    self.all_samples_r = self.update_variable(
                        self.all_samples_r,
                        jnp.expand_dims(jnp.stack((one_sample['r_sample'], one_sample['r_sample'])), axis=0),
                    )
                else:
                    self.all_samples_r = self.update_variable(self.all_samples_r, one_sample['r_sample'])
            else:
                self.all_samples_r = self.update_variable(self.all_samples_r, one_sample['r_sample'])

        if self.save_all_Bf_params:
            self.all_params_mixing_matrix_samples = self.update_variable(
                self.all_params_mixing_matrix_samples,
                jnp.expand_dims(one_sample['params_mixing_matrix_sample'], axis=0),
            )

    # def update_scam_step_size(self, carry, new_carry, iteration):
    #     """
    #     Update the SCAM step size for the Metropolis-Hastings step sampling of Bf and r

    #     Parameters
    #     ----------
    #     carry: dictionary
    #         dictionary carry from all_sampling_steps function
    #     new_carry: dictionary
    #         updated dictionary from all_sampling_steps function
    #     iteration: int
    #         current iteration number
    #     """
    #     total_number_iterations = iteration + self.number_iterations_done + 1

    #     # Update the SCAM step size for the Metropolis-Hastings step sampling of r
    #     new_carry['empirical_variance_r'] = get_1d_recursive_empirical_covariance(
    #         total_number_iterations,
    #         new_carry['r_sample'],
    #         carry['mean_r'],
    #         carry['empirical_variance_r'],
    #         s_param=self.s_param_scam,
    #         epsilon_param=self.epsilon_param_scam_r,
    #     ).squeeze()
    #     new_carry['mean_r'] = (total_number_iterations * carry['mean_r'] + new_carry['r_sample']) / (
    #         total_number_iterations + 1
    #     )

    #     # Update the SCAM step size for the Metropolis-Hastings step sampling of Bf
    #     new_carry['empirical_variance_Bf'] = get_1d_recursive_empirical_covariance(
    #         total_number_iterations,
    #         new_carry['params_mixing_matrix_sample'],
    #         carry['mean_Bf'],
    #         carry['empirical_variance_Bf'],
    #         s_param=self.s_param_scam,
    #         epsilon_param=self.epsilon_param_scam_Bf,
    #     )
    #     new_carry['mean_Bf'] = (total_number_iterations * carry['mean_Bf'] + new_carry['params_mixing_matrix_sample']) / (
    #         total_number_iterations + 1
    #     )

    def perform_Gibbs_sampling(
        self,
        input_freq_maps,
        c_ell_approx,
        CMB_c_ell,
        init_params_mixing_matrix,
        initial_guess_r=1e-8,
        initial_wiener_filter_term=None,
        initial_fluctuation_maps=None,
        theoretical_r0_total=None,
        theoretical_r1_tensor=None,
        **dictionnary_additional_parameters,
    ):
        r"""
        Perform sampling steps with:
            1. The sampling of \eta by computing \eta =  x + C_approx^(1/2) N_c^{-1/2} y ; where x is band-limited
            2. A CG for the Wiener filter (WF) and fluctuation variables s_c: (s_c - s_{c,WF})^t (C^{-1} + N_c^{-1}) (s_c - s_{c,WF})
            3. The c_ell sampling, either by parametrizing it by r or by sampling an inverse Wishart distribution
            4. Mixing matrix Bf sampling with: -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + \eta^t (Id + C_{approx}^{1/2} N_c^{-1} C_{approx}^{1/2}) \eta

        The results of the chain will be stored in the class attributes, depending if the save options are put to True or False:
            - self.all_samples_eta (if self.save_eta_chain_maps is True)
            - self.all_samples_wiener_filter_maps (if self.save_CMB_chain_maps is True)
            - self.all_samples_fluctuation_maps (if self.save_CMB_chain_maps is True)
            - self.all_samples_r (if self.sample_r_Metropolis is True)
            - self.all_samples_CMB_c_ell (if self.sample_C_inv_Wishart is True)
            - self.all_params_mixing_matrix_samples (always)

        This same function can be used to continue a chain from a previous run, by giving the number of iterations already done in the MicmacSampler object,
        giving the chains to the attributes of the object, and giving the last iteration results as initial guesses.

        Parameters
        ----------
        input_freq_maps: array[float] of dimensions [frequencies, nstokes, n_pix]
            input frequency maps
        c_ell_approx: array[float] of dimensions [number_correlations, lmax+1]
            approximate CMB power spectra for the latent parameter \eta defining the ad-hoc correction term
        CMB_c_ell: array[float] of dimensions [number_correlations, lmax+1]
            CMB power spectra, where number_correlations is the number of auto- and cross-correlations relevant considering the number of Stokes parameters
        init_params_mixing_matrix: array[float] of dimensions [len_params]
            initial parameters for the mixing matrix elements Bf; expected to be given flattened as [Bf_s1, Bf_s2, ..., Bf_sn, Bf_d1, ..., Bf_dn]
        initial_guess_r: float (optional)
            initial guess for r, default 1e-8
        initial_wiener_filter_term: array[float] of dimensions [nstokes, n_pix] or empty (optional)
            initial guess for the Wiener filter term, default empty array
        initial_fluctuation_maps: array[float] of dimensions [nstokes, n_pix] or empty (optional)
            initial guess for the fluctuation maps, default empty array
        theoretical_r0_total: array[float] of dimensions [number_correlations, lmax+1-lmin] (optional)
            theoretical reduced covariance matrix for the CMB scalar modes, default empty array
        theoretical_r1_tensor: array[float] of dimensions [number_correlations, lmax+1-lmin] (optional)
            theoretical reduced covariance matrix for the CMB tensor modes, default empty array
        dictionnary_additional_parameters: dictionary
            additional parameters to give to the function, currently only the ones related to the SCAM step size

        Notes
        -----
        The formalism relies on the ability to have an inverse for C_approx (even though it is never computed effectively in the code), and may lead to numerical instabilities if the C_approx matrix is not well-conditioned.
        """

        time_test = time.time()

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

        ## Testing the inverse frequency noise
        assert (
            self.freq_inverse_noise is not None
        ), 'The inverse noise for the frequencies should be provided as an attribute of the MicmacSampler object'
        assert self.freq_inverse_noise.shape == (
            self.n_frequencies,
            self.n_frequencies,
            self.n_pix,
        ), 'The inverse noise for the frequencies should have dimensions [n_frequencies,n_frequencies,n_pix]'

        ## Testing the initial WF term, or initialize it properly
        if initial_wiener_filter_term is None:
            wiener_filter_term = jnp.zeros((self.nstokes, self.n_pix))
        else:
            assert len(initial_wiener_filter_term.shape) == 2
            assert initial_wiener_filter_term.shape == (self.nstokes, self.n_pix)
            wiener_filter_term = initial_wiener_filter_term

        ## Testing the initial fluctuation term, or initialize it properly
        if initial_fluctuation_maps is None:
            fluctuation_maps = jnp.zeros((self.nstokes, self.n_pix))
        else:
            assert len(initial_fluctuation_maps.shape) == 2
            assert initial_fluctuation_maps.shape == (self.nstokes, self.n_pix)
            fluctuation_maps = initial_fluctuation_maps

        ## Testing the initial spectra given in case the sampling is done with r
        if self.sample_r_Metropolis:
            assert len(theoretical_r0_total.shape) == 2
            assert (
                theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin
            )  # thoertical_r0_total must cover multipoles [lmin,lmax]
            assert (
                theoretical_r1_tensor.shape == theoretical_r0_total.shape
            )  # theoretical_r1_tensor must cover the same multipoles as theoretical_r0_total [lmin,lmax]

            # Transforming into the reduced (red) format [lmax+1-lmin,nstokes,nstokes]
            theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
            theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)
            assert theoretical_red_cov_r0_total.shape[1] == self.nstokes

        ## Testing the initial CMB spectra and C_approx spectra given
        if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):
            CMB_c_ell = CMB_c_ell[
                indices_to_consider, :
            ]  # Selecting only the relevant auto- and cross-correlations for polarization
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[
                indices_to_consider, :
            ]  # Selecting only the relevant auto- and cross-correlations for polarization

        assert len(CMB_c_ell.shape) == 2
        assert CMB_c_ell.shape[1] == self.lmax + 1
        assert c_ell_approx.shape == CMB_c_ell.shape
        red_cov_approx_matrix = jnp.array(get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin :, ...])
        assert (
            jnp.linalg.det(red_cov_approx_matrix) != 0
        ).any(), 'The approximate covariance matrix should be invertible ; if you want to put it to zero, please put it instead to a very small value'

        ## Testing the initial mixing matrix
        if self.n_components != 1:
            assert init_params_mixing_matrix.shape == (
                self.len_params,
            ), 'The initial mixing matrix should have the same length as the number of parameters'

        ## Testing the input frequency maps
        assert input_freq_maps.shape == (
            self.n_frequencies,
            self.nstokes,
            self.n_pix,
        ), 'The input frequency maps should have dimensions [n_frequencies,nstokes,n_pix]'

        ## Testing the mask
        assert np.abs(self.mask).sum() != 0, 'The mask must not be entirely zero'

        ## Testing the initial guess for r
        assert np.size(initial_guess_r) == 1
        if self.below_0_min_r_value and self.min_r_value is None:
            print('Setting min_r_value so that C(r) is positive definite', flush=True)
            self.min_r_value = -np.min(theoretical_red_cov_r0_total[:, 1, 1] / theoretical_red_cov_r1_tensor[:, 1, 1])
        assert (
            initial_guess_r > self.min_r_value
        ), f'Not allowing first guess for r {initial_guess_r} to have value lower than min_r_value {self.min_r_value}'

        # Preparing for the full Gibbs sampling
        len_pos_special_freqs = len(self.pos_special_freqs)

        # if self.fwhm is not None:
        #     ell_range = jnp.arange(self.lmin, self.lmax + 1)
        #     spin = 2
        #     self.beam_harmonic = jnp.exp(-0.5 * (ell_range * (ell_range + 1) - spin**2)* self.fwhm**2 / (8 * np.log(2)))
        # else:
        #     self.beam_harmonic = jnp.ones(self.lmax + 1 - self.lmin)

        if self.use_binning:
            print('Using binning for the sampling of CMB covariance !!!', flush=True)
            print('Binning distribution:', self.bin_ell_distribution, flush=True)

        ## Initial guesses preparation

        ## eta
        initial_eta = jnp.zeros((self.nstokes, self.n_pix))
        ## CMB covariance preparation in the format [lmax,nstokes,nstokes]

        red_cov_matrix = get_reduced_matrix_from_c_ell(CMB_c_ell)[self.lmin :, ...]
        ## parameters of the mixing matrix
        params_mixing_matrix_init_sample = jnp.array(init_params_mixing_matrix, copy=True)

        # Preparing the sampling functions
        ## Function to sample eta
        func_logproba_eta = self.get_conditional_proba_correction_likelihood_JAX_v2d
        ## Function to compute the Wiener filter term
        sampling_func_WF = self.solve_generalized_wiener_filter_term_v2d
        ## Function to sample the fluctuation maps
        sampling_func_Fluct = self.get_fluctuating_term_maps_v2d
        ## Function to sample the CMB covariance from inverse Wishart
        func_get_inverse_wishart_sampling_from_c_ells = self.get_inverse_wishart_sampling_from_c_ells
        if self.use_binning:
            func_get_inverse_wishart_sampling_from_c_ells = self.get_binned_inverse_wishart_sampling_from_c_ells_v3
        ## Function to sample the CMB covariance parametrize from r
        r_sampling_MH = single_Metropolis_Hasting_step
        # r_sampling_MH = bounded_single_Metropolis_Hasting_step
        if self.sample_r_Metropolis:
            log_proba_r = self.get_conditional_proba_C_from_r_wBB
            if self.use_alm_sampling_r or self.use_alm_sampling_r_wEE:
                assert self.use_alm_sampling_r_wEE != self.use_alm_sampling_r
            if self.use_alm_sampling_r:
                log_proba_r = self.get_conditional_proba_C_walm_from_r_wBB
            if self.use_alm_sampling_r_wEE:
                log_proba_r = self.get_conditional_proba_C_walm_from_r
            if self.use_binning:
                print('Using BB binning for the sampling of r !!!', flush=True)
                log_proba_r = self.get_binned_conditional_proba_C_from_r_wBB

        ## Function to sample the mixing matrix free parameters in the most general way
        jitted_Bf_func_sampling = jax.jit(
            self.get_conditional_proba_mixing_matrix_v2b_JAX, static_argnames=['biased_bool']
        )
        sampling_func = separate_single_MH_step_index_accelerated

        if self.biased_version or self.perturbation_eta_covariance:
            print('Using biased version or perturbation version of mixing matrix sampling !!!', flush=True)

            assert (
                self.biased_version != self.perturbation_eta_covariance
            ), 'Cannot use both biased and perturbation version of mixing matrix sampling !!!'

            ## Function to sample the mixing matrix free parameters through the difference of the log-proba, to have only one CG done
            jitted_Bf_func_sampling = jax.jit(
                self.get_conditional_proba_mixing_matrix_v3_JAX,
                static_argnames=['biased_bool', 'use_mask_contribution_eta'],
            )
            sampling_func = separate_single_MH_step_index_v2b

            if self.use_alternative_Bf_sampling:
                print('Using alternative version of the Bf sampling !!!', flush=True)
                jitted_Bf_func_sampling = jax.jit(
                    self.get_conditional_proba_mixing_matrix_v4_JAX,
                    static_argnames=['biased_bool'],
                )

            if self.simultaneous_accept_rate:
                ## More efficient version of the mixing matrix sampling

                ## MH step function to sample the mixing matrix free parameters with patches simultaneous computed accept rate
                print('Using simultaneous accept rate version of mixing matrix sampling !!!', flush=True)
                print(
                    '---- ATTENTION: This assumes all patches are distributed in the same way for all parameters !',
                    flush=True,
                )
                jitted_Bf_func_sampling = jax.jit(
                    self.get_conditional_proba_mixing_matrix_v3_pixel_JAX,
                    static_argnames=['biased_bool'],
                )
                sampling_func = separate_single_MH_step_index_v4_pixel
                if (self.size_patches != self.size_patches[0]).any():
                    sampling_func = separate_single_MH_step_index_v4b_pixel
                    # raise NotImplemented("All patches should have the same size for the simultaneous accept rate version of mixing matrix sampling for now !!!")

                ## Redefining the free Bf indexes to sample to the one
                # condition_unobserved_patches = self.get_cond_unobserved_patches() ## Get boolean array to identify which free indexes are not relevant
                # print("Previous free indexes for Bf", self.indexes_free_Bf, flush=True)
                # self.indexes_free_Bf = jnp.array(self.indexes_free_Bf).at[condition_unobserved_patches].get()
                # print("New free indexes for Bf", self.indexes_free_Bf, flush=True)

                print('Previous free indexes for Bf', self.indexes_free_Bf, self.indexes_free_Bf.size, flush=True)
                self.indexes_free_Bf = self.indexes_free_Bf.at[
                    self.get_cond_unobserved_patches_from_indices_optimized(self.indexes_free_Bf)
                ].get()
                ## Get boolean array to identify which free indexes are not relevant
                print('New free indexes for Bf', self.indexes_free_Bf, self.indexes_free_Bf.size, flush=True)

                indexes_patches_Bf = jnp.array(self.indexes_b.ravel(order='F'), dtype=jnp.int64)

                def which_interval(carry, index_Bf):
                    """
                    Selecting the patches to be used for the Bf sampling by checking if the index_Bf is in the interval of the patches
                    """
                    return (
                        carry
                        | ((index_Bf >= indexes_patches_Bf) & (index_Bf < indexes_patches_Bf + self.size_patches)),
                        index_Bf,
                    )

                condition, _ = jlax.scan(
                    which_interval, jnp.zeros_like(self.size_patches, dtype=bool), self.indexes_free_Bf
                )

                first_indices_patches_free_Bf = indexes_patches_Bf[condition]
                max_len_patches_Bf = int(np.max(self.size_patches[condition]))
                size_patches = self.size_patches[condition]

        ## Preparing minmum value of r sampling

        ## Preparing the random JAX PRNG key
        if np.size(self.seed) == 1:
            PRNGKey = random.PRNGKey(self.seed)
        elif np.size(self.seed) == 2:
            PRNGKey = jnp.array(self.seed, dtype=jnp.uint32)
        else:
            raise ValueError('Seed should be either a scalar or a 2D array interpreted as a JAX PRNG Key!')

        ## Computing the number of iterations to perform
        actual_number_of_iterations = self.number_iterations_sampling  # - self.number_iterations_done

        if not (self.classical_Gibbs):
            ## Preparing the step-size for Metropolis-within-Gibbs of Bf sampling

            ## try/except step only because jsp.linalg.sqrtm is not implemented in GPU
            try:
                initial_step_size_Bf = jnp.array(jnp.diag(jsp.linalg.sqrtm(self.covariance_Bf)), dtype=jnp.float64)
            except:
                initial_step_size_Bf = jnp.array(jnp.diag(jnp.sqrt(self.covariance_Bf)), dtype=jnp.float64)
            assert len(initial_step_size_Bf.shape) == 1
            print('Step-size Bf', initial_step_size_Bf, flush=True)
            if self.covariance_Bf.shape[0] != self.len_params:
                print('Covariance matrix for Bf is not of the right shape !', flush=True)
                # initial_step_size_Bf = jnp.repeat(initial_step_size_Bf, self.len_params//self.covariance_Bf.shape[0], axis=0)
                if self.covariance_Bf.shape[0] != 2 * (self.n_frequencies - len_pos_special_freqs):
                    raise ValueError(
                        f'Covariance matrix for Bf is not of the right shape with shape {self.covariance_Bf.shape[0]}, it cannot be properly expanded with the considered multipatch distribution!'
                    )

                if (
                    self.size_patches is not None and (self.size_patches == self.size_patches[0]).all()
                ):  # If all patches have the same size
                    initial_step_size_Bf = jnp.broadcast_to(  # Broadcasting the step-size for each patch size
                        initial_step_size_Bf,
                        (self.len_params // self.covariance_Bf.shape[0], self.covariance_Bf.shape[0]),
                    ).ravel(order='F')
                else:  # If patches have different sizes
                    previous_initial_Bf = jnp.copy(initial_step_size_Bf)
                    initial_step_size_Bf = jnp.zeros(self.len_params)
                    number_free_Bf = (self.n_frequencies - len_pos_special_freqs) * (self.n_components - 1)

                    extended_array = np.zeros((number_free_Bf + 1), dtype=np.int64)
                    extended_array[0] = 0
                    extended_array[1:] = self.sum_size_patches_indexed_freq_comp.ravel(order='F') + self.size_patches

                    for i in range(
                        self.size_patches.size
                    ):  # Loop over the patches to update the step-size for each patch size
                        initial_step_size_Bf = initial_step_size_Bf.at[extended_array[i] : extended_array[i + 1]].set(
                            previous_initial_Bf[i]
                        )
                    initial_step_size_Bf = initial_step_size_Bf.at[extended_array[-1] :].set(previous_initial_Bf[-1])

                print('New step-size Bf', initial_step_size_Bf, flush=True)

        ## Few prints to re-check the toml parameters chosen
        if self.classical_Gibbs:
            print('Not sampling for eta and Bf, only for s_c and the CMB covariance!', flush=True)
        if self.sample_r_Metropolis:
            print('Sample for r instead of C!', flush=True)
            if self.use_alm_sampling_r:
                print('Sample for r with alms!', flush=True)
            if self.use_alm_sampling_r_wEE:
                print('Sample for r with alms and EE!', flush=True)
            if self.limit_r_value:
                print(f'Limiting the r value to be superior to {self.min_r_value} !', flush=True)
        if self.non_centered_moves:
            print('Using non-centered moves for C sampling !', flush=True)
            if self.save_intermediary_centered_moves:
                print('Saving intermediary centered moves for C sampling !', flush=True)
        else:
            print('Sample for C with inverse Wishart !', flush=True)

        # Few steps to improve the speed of the code

        ## Preparing the square root matrix of C_approx
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)

        ## Preparing the preconditioner in the case of a full sky and white noise
        use_precond = False
        if self.mask.sum() == self.n_pix and self.freq_noise_c_ell is not None:
            assert len(self.freq_noise_c_ell.shape) == 3
            assert self.freq_noise_c_ell.shape[0] == self.n_frequencies
            assert self.freq_noise_c_ell.shape[1] == self.n_frequencies
            assert (self.freq_noise_c_ell.shape[2] == self.lmax + 1) or (
                self.freq_noise_c_ell.shape[2] == self.lmax + 1 - self.lmin
            )
            if self.freq_noise_c_ell.shape[2] == self.lmax + 1:
                self.freq_noise_c_ell = self.freq_noise_c_ell[..., self.lmin :]
            self.freq_noise_c_ell = jnp.array(self.freq_noise_c_ell)

            print('Full sky case, use_precond !', flush=True)
            use_precond = True

        ## Finally starting the Gibbs sampling !!!
        print(
            f'Starting {self.number_iterations_sampling} iterations in addition to {self.number_iterations_done} iterations done',
            flush=True,
        )

        def wrapper_map2alm(maps_, lmax=self.lmax, n_iter=self.n_iter, nside=self.nside):
            maps_np = jax.tree.map(np.asarray, maps_).reshape((3, 12 * nside**2))
            alm_T, alm_E, alm_B = hp.map2alm(maps_np, lmax=lmax, iter=n_iter)
            return np.array([alm_T, alm_E, alm_B])

        ## Preparing JAX pure call back for the Healpy map2alm function
        @partial(jax.jit, static_argnums=(1))
        def pure_call_map2alm(maps_, lmax):
            shape_output = (
                3,
                (lmax + 1) * (lmax // 2 + 1),
            )  ## Shape of the output alms : [3 for all Stokes params, (lmax+1)*(lmax+2)//2 for all alms in the Healpy convention]
            return jax.pure_callback(wrapper_map2alm, jax.ShapeDtypeStruct(shape_output, np.complex128), maps_.ravel())

        @scan_tqdm(
            actual_number_of_iterations,
        )
        def all_sampling_steps(carry, iteration):
            """
            1-step Gibbs sampling function, performing the following:
            - Sampling of eta, for the correction term ; perform as well a CG if the perturbation approach is chosen
            - Sampling of s_c, for the constrained CMB map realization ; sampling both Wiener filter and fluctuation maps
            - Sampling of C or r parametrizing C, for the CMB covariance matrix
            - Sampling of the free Bf parameters, for the mixing matrix

            Parameters
            ----------
            carry: dictionary
                dictionary containing the following variables at 1 iteration depending on the option chosen: WF maps, fluctuation maps, CMB covariance, r samples, Bf samples, PRNGKey
            iteration: int
                current iteration number

            Returns
            -------
            new_carry: dictionary
                dictionary containing the following variables at the next iteration: WF maps, fluctuation maps, CMB covariance, r sample, Bf sample, PRNGKey
            all_samples: dictionary
                dictionary containing the variables to save as chains, so depending on the options chosen: eta maps, WF maps, fluctuation maps, CMB covariance, r sample, Bf sample
            """

            # Extracting the JAX PRNG key from the carry
            PRNGKey = carry['PRNGKey']

            # Preparing the new carry and all_samples to save the chains
            new_carry = dict()
            all_samples = dict()

            # Preparing a new PRNGKey for eta sampling
            PRNGKey, subPRNGKey = random.split(PRNGKey)

            # Extracting the mixing matrix parameters and initializing the new one
            # self.update_params(carry['params_mixing_matrix_sample'])
            # mixing_matrix_sampled = self.get_B(jax_use=True)
            mixing_matrix_sampled = self.get_B_from_params(carry['params_mixing_matrix_sample'], jax_use=True)

            # Few checks for the mixing matrix
            chx.assert_shape(mixing_matrix_sampled, (self.n_frequencies, self.n_components, self.n_pix))

            # Application of new mixing matrix to the noise covariance and extracted CMB map from data
            invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, mixing_matrix_sampled, jax_use=True)
            BtinvN_sqrt = get_BtinvN(jnp.sqrt(self.freq_inverse_noise), mixing_matrix_sampled, jax_use=True)
            s_cML = get_Wd(self.freq_inverse_noise, mixing_matrix_sampled, input_freq_maps, jax_use=True)[0]

            # Sampling step 1: sampling of Gaussian variable eta

            ## Initialize the preconditioner for the eta contribution
            precond_func_eta = None

            ## Sampling of eta if not using the classical Gibbs sampling and neither the biased version
            if not (self.classical_Gibbs) and not (self.biased_version):
                # Preparing random variables
                map_random_x = None
                map_random_y = None

                # Sampling eta maps
                new_carry['eta_maps'] = self.get_sampling_eta_v2(
                    red_cov_approx_matrix_sqrt,
                    invBtinvNB,
                    BtinvN_sqrt,
                    subPRNGKey,
                    map_random_x=map_random_x,
                    map_random_y=map_random_y,
                    suppress_low_modes=self.suppress_low_modes,
                )

                # Checking shape of the resulting maps
                chx.assert_shape(new_carry['eta_maps'], (self.nstokes, self.n_pix))

                # Preparing the preconditioner for the CG
                if use_precond:
                    ## Assuming a harmonic noise with the pixel average of the mixing matrix
                    noise_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled.mean(axis=2))[0, 0]
                    ## Getting N_c^{-1} for the harmonic noise covariance
                    red_inv_noise_c_ell = jnp.linalg.pinv(
                        get_reduced_matrix_from_c_ell_jax(
                            jnp.stack([noise_c_ell, noise_c_ell, jnp.zeros_like(noise_c_ell)])
                        )
                    )
                    red_preconditioner_eta = jnp.linalg.pinv(
                        jnp.eye(self.nstokes)
                        + jnp.einsum(
                            'lij,ljk,lkm->lim',
                            red_cov_approx_matrix_sqrt,
                            red_inv_noise_c_ell,
                            red_cov_approx_matrix_sqrt,
                        )
                    )
                    precond_func_eta = lambda x: maps_x_red_covariance_cell_JAX(
                        x.reshape((self.nstokes, self.n_pix)),
                        red_preconditioner_eta,
                        nside=self.nside,
                        lmin=self.lmin,
                        n_iter=self.n_iter,
                    ).ravel()

                if self.perturbation_eta_covariance:
                    # Computing the inverse associated log proba term fixed correction covariance for the Bf sampling, in case of the perturbative approach
                    _, inverse_term = func_logproba_eta(
                        invBtinvNB[0, 0, ...],
                        new_carry['eta_maps'],
                        red_cov_approx_matrix_sqrt,
                        first_guess=carry['inverse_term'],
                        return_inverse=True,
                        precond_func=precond_func_eta,
                    )
                else:
                    inverse_term = carry['inverse_term']

                if self.save_eta_chain_maps:
                    all_samples['eta_maps'] = new_carry['eta_maps']

            # Sampling step 2: sampling of Gaussian variable s_c, contrained CMB map realization

            ## Geting the square root matrix of the sampled CMB covariance
            red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(carry['red_cov_matrix_sample'])

            # Preparing the preconditioner to use for the sampling of the CMB maps
            precond_func_s_c = None
            if use_precond:
                ## Assuming a harmonic noise with the pixel average of the mixing matrix
                noise_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled.mean(axis=2))[0, 0]
                ## Getting N_c^{-1} for the harmonic noise covariance
                red_inv_noise_c_ell = jnp.linalg.pinv(
                    get_reduced_matrix_from_c_ell_jax(
                        jnp.stack([noise_c_ell, noise_c_ell, jnp.zeros_like(noise_c_ell)])
                    )
                )  # [self.lmin:]
                red_preconditioner_s_c = jnp.linalg.pinv(
                    jnp.eye(self.nstokes)
                    + jnp.einsum('lij,ljk,lkm->lim', red_cov_matrix_sqrt, red_inv_noise_c_ell, red_cov_matrix_sqrt)
                )

                precond_func_s_c = lambda x: maps_x_red_covariance_cell_JAX(
                    x.reshape((self.nstokes, self.n_pix)),
                    red_preconditioner_s_c,
                    nside=self.nside,
                    lmin=self.lmin,
                    n_iter=self.n_iter,
                ).ravel()

            ## Computing an initial guess closer to the actual start of the CG for the Wiener filter
            initial_guess_WF = maps_x_red_covariance_cell_JAX(
                carry['wiener_filter_term'],
                jnp.linalg.pinv(red_cov_matrix_sqrt),
                nside=self.nside,
                lmin=self.lmin,
                n_iter=self.n_iter,
            )
            ## Sampling the Wiener filter term
            new_carry['wiener_filter_term'] = sampling_func_WF(
                s_cML, red_cov_matrix_sqrt, invBtinvNB, initial_guess=initial_guess_WF, precond_func=precond_func_s_c
            )

            ## Preparing the random variables for the fluctuation term
            PRNGKey, new_subPRNGKey = random.split(PRNGKey)
            map_random_realization_xi = None
            map_random_realization_chi = None

            ## Getting the fluctuation maps terms, for the variance of the variable s_c
            initial_guess_Fluct = maps_x_red_covariance_cell_JAX(
                carry['fluctuation_maps'],
                jnp.linalg.pinv(red_cov_matrix_sqrt),
                nside=self.nside,
                lmin=self.lmin,
                n_iter=self.n_iter,
            )
            ## Sampling the fluctuation maps
            new_carry['fluctuation_maps'] = sampling_func_Fluct(
                red_cov_matrix_sqrt,
                invBtinvNB,
                BtinvN_sqrt,
                new_subPRNGKey,
                map_random_realization_xi=map_random_realization_xi,
                map_random_realization_chi=map_random_realization_chi,
                initial_guess=initial_guess_Fluct,
                precond_func=precond_func_s_c,
            )

            ## Constructing the sampled CMB map
            s_c_sample = new_carry['fluctuation_maps'] + new_carry['wiener_filter_term']

            if self.save_CMB_chain_maps:
                ## Saving the sampled Wiener filter term and fluctuation maps if chosen to
                all_samples['wiener_filter_term'] = new_carry['wiener_filter_term']
                all_samples['fluctuation_maps'] = new_carry['fluctuation_maps']

            ## Checking the shape of the resulting maps
            chx.assert_shape(new_carry['wiener_filter_term'], (self.nstokes, self.n_pix))
            chx.assert_shape(new_carry['fluctuation_maps'], (self.nstokes, self.n_pix))
            chx.assert_shape(s_c_sample, (self.nstokes, self.n_pix))

            # Sampling step 3: sampling of CMB covariance C

            ## Preparing the c_ell which will be used for the sampling
            c_ells_Wishart_ = get_cell_from_map_jax(s_c_sample, lmax=self.lmax, n_iter=self.n_iter)[:, self.lmin :]

            ## Saving the corresponding spectrum
            if self.save_s_c_spectra:
                all_samples['s_c_spectra'] = c_ells_Wishart_

            # ### Getting them in the format [lmax,nstokes,nstokes] multiplied by 2 ell+1, to take into account the m
            # red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_*(2*jnp.arange(self.lmax+1) + 1))

            ### Getting them in the format [lmax,nstokes,nstokes] without the facor 2 ell+1 to take into account the m
            red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_)

            ## Preparing the new PRNGkey
            PRNGKey, new_subPRNGKey_2 = random.split(PRNGKey)

            ## Performing the sampling
            if self.sample_C_inv_Wishart:
                # Sampling C with inverse Wishart
                new_carry['red_cov_matrix_sample'] = func_get_inverse_wishart_sampling_from_c_ells(
                    c_ells_Wishart_,
                    PRNGKey=new_subPRNGKey_2,
                    old_sample=carry['red_cov_matrix_sample'],
                    acceptance_posdef=self.acceptance_posdef,
                )
                all_samples['red_cov_matrix_sample'] = new_carry['red_cov_matrix_sample']

            elif self.sample_r_Metropolis:
                # Sampling r which will parametrize C(r) = C_scalar + r*C_tensor

                step_size_r = self.step_size_r

                if self.use_scam_step_size:
                    # step_size_r = jnp.where(iteration > self.burn_in_scam, jnp.sqrt(self.s_param_scam*(carry['empirical_variance_r'] + self.epsilon_param_scam_r)), self.step_size_r)
                    # step_size_r = jnp.where(
                    #     iteration > self.burn_in_scam, jnp.sqrt(carry['empirical_variance_r']), self.step_size_r
                    # )

                    step_size_r = jnp.sqrt(carry['empirical_variance_r'])

                    all_samples['empirical_variance_r'] = step_size_r**2
                    all_samples['mean_r'] = carry['mean_r']

                dictionary_arguments_sampling_r = {
                    'random_PRNGKey': new_subPRNGKey_2,
                    'old_sample': carry['r_sample'],
                    'step_size': step_size_r,
                    'log_proba': log_proba_r,
                    'theoretical_red_cov_r1_tensor': jnp.copy(theoretical_red_cov_r1_tensor),
                    'theoretical_red_cov_r0_total': jnp.copy(theoretical_red_cov_r0_total),
                }

                if self.use_alm_sampling_r:
                    s_c_sample_extended = jnp.vstack((jnp.zeros_like(s_c_sample[0]), s_c_sample))

                    # alm_s_c = jnp.expand_dims(pure_call_map2alm(s_c_sample_extended, lmax=self.lmax)[2, ...], axis=0)
                    alm_s_c = pure_call_map2alm(s_c_sample_extended, lmax=self.lmax)[2, ...]

                    dictionary_arguments_sampling_r['alm_s_c'] = alm_s_c

                elif self.use_alm_sampling_r_wEE:
                    s_c_sample_extended = jnp.vstack((jnp.zeros_like(s_c_sample[0]), s_c_sample))

                    alm_s_c = pure_call_map2alm(s_c_sample_extended, lmax=self.lmax)[1:, ...]

                    dictionary_arguments_sampling_r['alm_s_c'] = alm_s_c
                else:
                    dictionary_arguments_sampling_r['lmin_BB'] = self.lmin_BB
                    dictionary_arguments_sampling_r['red_sigma_ell'] = red_c_ells_Wishart_modified

                    if self.lmin_BB is not None:
                        dictionary_arguments_sampling_r[
                            'theoretical_red_cov_r1_tensor'
                        ] = theoretical_red_cov_r1_tensor[self.lmin_BB - self.lmin :, ...]
                        dictionary_arguments_sampling_r['theoretical_red_cov_r0_total'] = theoretical_red_cov_r0_total[
                            self.lmin_BB - self.lmin :, ...
                        ]
                        dictionary_arguments_sampling_r['red_sigma_ell'] = red_c_ells_Wishart_modified.at[
                            self.lmin_BB - self.lmin :, ...
                        ].get()

                new_carry['r_sample'] = r_sampling_MH(**dictionary_arguments_sampling_r)

                if self.limit_r_value:
                    new_carry['r_sample'] = jnp.where(
                        new_carry['r_sample'] < self.min_r_value, carry['r_sample'], new_carry['r_sample']
                    )

                ## Reconstructing the new spectra from r
                new_carry['red_cov_matrix_sample'] = (
                    theoretical_red_cov_r0_total + new_carry['r_sample'] * theoretical_red_cov_r1_tensor
                )

                ## Binning if needed
                if self.use_binning:
                    new_carry['red_cov_matrix_sample'] = self.bin_and_reproject_red_c_ell(
                        new_carry['red_cov_matrix_sample']
                    )

                ## Saving the r sample
                all_samples['r_sample'] = new_carry['r_sample']
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')

            if self.non_centered_moves:
                PRNGKey, new_subPRNGKey_2b = random.split(PRNGKey)
                if self.sample_r_Metropolis:
                    new_r_sample = r_sampling_MH(
                        random_PRNGKey=new_subPRNGKey_2b,
                        old_sample=new_carry['r_sample'],
                        step_size=self.step_size_r,
                        log_proba=self.get_log_proba_non_centered_move_C_from_r,
                        old_r_sample=new_carry['r_sample'],
                        invBtinvNB=invBtinvNB,
                        s_cML=s_cML,
                        s_c_sample=s_c_sample,
                        theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor,
                        theoretical_red_cov_r0_total=theoretical_red_cov_r0_total,
                    )
                    # min_value=self.min_r_to_sample)

                    if self.limit_r_value:
                        new_r_sample = jnp.where(new_r_sample < self.min_r_value, new_carry['r_sample'], new_r_sample)

                    new_carry['red_cov_matrix_sample'] = (
                        theoretical_red_cov_r0_total + new_r_sample * theoretical_red_cov_r1_tensor
                    )

                    if self.save_intermediary_centered_moves:
                        all_samples['r_sample'] = jnp.stack((new_carry['r_sample'], new_r_sample))
                    else:
                        all_samples['r_sample'] = new_r_sample

                    new_carry['r_sample'] = new_r_sample

            ## Checking the shape of the resulting covariance matrix, and correcting it if needed
            if new_carry['red_cov_matrix_sample'].shape[0] == self.lmax + 1:
                new_carry['red_cov_matrix_sample'] = new_carry['red_cov_matrix_sample'][self.lmin :]

            ## Small check on the shape of the resulting covariance matrix
            chx.assert_shape(
                new_carry['red_cov_matrix_sample'], (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes)
            )

            # Sampling step 4: sampling of mixing matrix Bf

            ## Preparation of sampling step 4

            ## First preparing the term: d - B_c s_c
            # full_data_without_CMB = input_freq_maps - jnp.broadcast_to(
            #     s_c_sample, (self.n_frequencies, self.nstokes, self.n_pix)
            # )
            full_data_without_CMB = input_freq_maps - jnp.broadcast_to(
                s_c_sample, (self.n_frequencies, self.nstokes, self.n_pix)
            )  # TODO: Not removing CMB from the data at this stage
            chx.assert_shape(full_data_without_CMB, (self.n_frequencies, self.nstokes, self.n_pix))

            ## Preparing the new PRNGKey
            PRNGKey, new_subPRNGKey_3 = random.split(PRNGKey)

            ## Performing the sampling
            if not (self.classical_Gibbs):
                # Preparing the step-size
                step_size_Bf = initial_step_size_Bf

                if self.use_scam_step_size:
                    # step_size_Bf = jnp.where(iteration > self.burn_in_scam, jnp.sqrt(self.s_param_scam *(carry['empirical_variance_Bf'] + self.epsilon_param_scam_Bf)), initial_step_size_Bf)
                    # step_size_Bf = jnp.where(
                    #     iteration > self.burn_in_scam, jnp.sqrt(carry['empirical_variance_Bf']), initial_step_size_Bf
                    # )
                    step_size_Bf = jnp.sqrt(carry['empirical_variance_Bf'])

                    # all_samples['empirical_variance_Bf'] = step_size_Bf
                    all_samples['empirical_variance_Bf'] = carry['empirical_variance_Bf']
                    all_samples['mean_Bf'] = carry['mean_Bf']

                # Sampling Bf
                if self.perturbation_eta_covariance or self.biased_version:
                    ## Preparing the parameters to provide for the sampling of Bf

                    if not (self.use_alternative_Bf_sampling):
                        dict_parameters_sampling_Bf = {
                            'indexes_Bf': self.indexes_free_Bf,
                            'full_data_without_CMB': full_data_without_CMB,
                            'red_cov_approx_matrix_sqrt': red_cov_approx_matrix_sqrt,
                            'old_params_mixing_matrix': carry['params_mixing_matrix_sample'],
                            'biased_bool': self.biased_version,
                            'use_mask_contribution_eta': self.use_mask_contribution_eta,
                        }
                    else:
                        dict_parameters_sampling_Bf = {
                            'indexes_Bf': self.indexes_free_Bf,
                            'input_data': jnp.copy(input_freq_maps),
                            's_c_sample': s_c_sample,
                            'red_cov_approx_matrix_sqrt': red_cov_approx_matrix_sqrt,
                            'old_params_mixing_matrix': carry['params_mixing_matrix_sample'],
                            'biased_bool': self.biased_version,
                            'use_mask_contribution_eta': self.use_mask_contribution_eta,
                        }
                    if self.perturbation_eta_covariance:
                        ## Precomputing the term C_approx^{1/2} A^{-1} eta = C_approx^{1/2} ( Id + C_approx^{1/2} N_{c,old}^{-1} C_approx^{1/2} )^{-1} eta
                        inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(
                            inverse_term.reshape(self.nstokes, self.n_pix),
                            red_cov_approx_matrix_sqrt,
                            nside=self.nside,
                            lmin=self.lmin,
                            n_iter=self.n_iter,
                        ).ravel()
                        dict_parameters_sampling_Bf['previous_inverse_x_Capprox_root'] = inverse_term_x_Capprox_root
                        dict_parameters_sampling_Bf['first_guess'] = inverse_term

                    if not (self.biased_version):
                        ## If not biased, provide the eta maps
                        dict_parameters_sampling_Bf['component_eta_maps'] = new_carry['eta_maps']
                    if self.simultaneous_accept_rate:
                        ## Provide as well the indexes of the patches in case of the uncorrelated patches version
                        dict_parameters_sampling_Bf['size_patches'] = size_patches
                        dict_parameters_sampling_Bf['max_len_patches_Bf'] = max_len_patches_Bf
                        dict_parameters_sampling_Bf['indexes_patches_Bf'] = first_indices_patches_free_Bf
                        dict_parameters_sampling_Bf['len_indexes_Bf'] = self.len_params
                        # TODO: Accelerate by removing indexes of indexes_patches_Bf if the corresponding patches are not in indexes_free_Bf, nor in the mask
                    ## Sampling Bf !
                    new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'] = sampling_func(
                        random_PRNGKey=new_subPRNGKey_3,
                        old_sample=carry['params_mixing_matrix_sample'],
                        step_size=step_size_Bf,
                        log_proba=jitted_Bf_func_sampling,
                        **dict_parameters_sampling_Bf,
                    )
                else:
                    ## Sampling Bf with older version -> might be slower
                    new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'], inverse_term = sampling_func(
                        random_PRNGKey=new_subPRNGKey_3,
                        old_sample=carry['params_mixing_matrix_sample'],
                        step_size=step_size_Bf,
                        indexes_Bf=self.indexes_free_Bf,
                        log_proba=jitted_Bf_func_sampling,
                        full_data_without_CMB=full_data_without_CMB,
                        component_eta_maps=new_carry['eta_maps'],
                        red_cov_approx_matrix_sqrt=red_cov_approx_matrix_sqrt,
                        first_guess=carry['inverse_term'],
                        biased_bool=self.biased_version,
                        precond_func=precond_func_eta,
                    )
                if self.perturbation_eta_covariance:
                    ## Passing the inverse term to the next iteration
                    new_carry['inverse_term'] = inverse_term

                # Checking the shape of the resulting mixing matrix
                chx.assert_shape(new_carry['params_mixing_matrix_sample'], (self.len_params,))
            else:
                ## Classical Gibbs sampling, no need to sample Bf but still needs to provide them to the next iteration in case it is used for the CMB noise component
                new_carry['params_mixing_matrix_sample'] = carry['params_mixing_matrix_sample']
                # all_samples['params_mixing_matrix_sample'] = new_carry['params_mixing_matrix_sample']

            ## Saving the Bf obtained
            all_samples['params_mixing_matrix_sample'] = new_carry['params_mixing_matrix_sample']

            # Updating the step-size in case of SCAM for the Metropolis-Hastings step
            if self.use_scam_step_size:
                ## Using the SCAM step-size for the Metropolis-Hasting step
                # new_carry = self.update_scam_step_size(carry, new_carry, iteration)
                total_number_iterations = (
                    iteration + self.number_iterations_done + 1 - self.burn_in_scam // self.scam_iteration_updates
                )

                update_scam_step_size = jnp.logical_and(
                    total_number_iterations > 0, total_number_iterations % self.scam_iteration_updates == 0
                )

                # Update the SCAM step size for the Metropolis-Hastings step sampling of r
                # new_carry['empirical_variance_r'] = get_1d_recursive_empirical_covariance(
                #     total_number_iterations,
                #     new_carry['r_sample'],
                #     carry['mean_r'],
                #     carry['empirical_variance_r'],
                #     s_param=self.s_param_scam,
                #     epsilon_param=self.epsilon_param_scam_r,
                # ).squeeze()
                # new_carry['mean_r'] = (total_number_iterations * carry['mean_r'] + carry['r_sample']) / (
                #     total_number_iterations + 1
                # )

                new_carry['empirical_variance_r'] = jax.lax.cond(
                    update_scam_step_size,
                    lambda x: get_1d_recursive_empirical_covariance(
                        total_number_iterations,
                        new_carry['r_sample'],
                        carry['mean_r'],
                        x,
                        s_param=self.s_param_scam,
                        epsilon_param=self.epsilon_param_scam_r,
                    ).squeeze(),
                    lambda x: x,
                    carry['empirical_variance_r'],
                )
                new_carry['mean_r'] = jax.lax.cond(
                    update_scam_step_size,
                    lambda x: (total_number_iterations * carry['mean_r'] + x) / (total_number_iterations + 1),
                    lambda x: x,
                    new_carry['r_sample'],
                )

                # Update the SCAM step size for the Metropolis-Hastings step sampling of Bf
                # new_carry['empirical_variance_Bf'] = get_1d_recursive_empirical_covariance(
                #     total_number_iterations,
                #     new_carry['params_mixing_matrix_sample'],
                #     carry['mean_Bf'],
                #     carry['empirical_variance_Bf'],
                #     s_param=self.s_param_scam,
                #     epsilon_param=self.epsilon_param_scam_Bf,
                # )
                # new_carry['mean_Bf'] = (
                #     total_number_iterations * carry['mean_Bf'] + carry['params_mixing_matrix_sample']
                # ) / (total_number_iterations + 1)

                new_carry['empirical_variance_Bf'] = jax.lax.cond(
                    update_scam_step_size,
                    lambda x: get_1d_recursive_empirical_covariance(
                        total_number_iterations,
                        new_carry['params_mixing_matrix_sample'],
                        carry['mean_Bf'],
                        x,
                        s_param=self.s_param_scam,
                        epsilon_param=self.epsilon_param_scam_Bf,
                    ),
                    lambda x: x,
                    carry['empirical_variance_Bf'],
                )
                new_carry['mean_Bf'] = jax.lax.cond(
                    update_scam_step_size,
                    lambda x: (total_number_iterations * carry['mean_Bf'] + x) / (total_number_iterations + 1),
                    lambda x: x,
                    new_carry['params_mixing_matrix_sample'],
                )

            ## Passing as well the PRNGKey to the next iteration
            new_carry['PRNGKey'] = PRNGKey
            return new_carry, all_samples

        ## Preparing the initial carry
        initial_carry = {
            'wiener_filter_term': wiener_filter_term,
            'fluctuation_maps': fluctuation_maps,
            'red_cov_matrix_sample': red_cov_matrix,
            'params_mixing_matrix_sample': params_mixing_matrix_init_sample,
            'PRNGKey': PRNGKey,
        }

        if not (self.classical_Gibbs) and not (self.biased_version):
            initial_carry['eta_maps'] = initial_eta
        if not (self.classical_Gibbs) and not (self.biased_version):
            initial_carry['inverse_term'] = jnp.zeros_like(initial_eta)
        if self.sample_r_Metropolis:
            initial_carry['r_sample'] = initial_guess_r
        if self.save_s_c_spectra:
            self.all_samples_s_c_spectra = self.update_variable(
                self.all_samples_s_c_spectra,
                jnp.expand_dims(jnp.zeros((self.n_correlations, self.lmax + 1 - self.lmin)), axis=0),
            )

        ## Initialising the first carry to the chains saved
        self.update_one_sample(initial_carry)

        print(
            '###### Time before entering scan and all_sampling_steps',
            (time.time() - time_test) / 60,
            'minutes',
            flush=True,
        )

        if self.use_scam_step_size:
            initial_carry['empirical_variance_r'] = jnp.array(self.step_size_r) ** 2
            initial_carry['empirical_variance_Bf'] = initial_step_size_Bf**2
            initial_carry['mean_r'] = jnp.array(initial_guess_r)
            initial_carry['mean_Bf'] = jnp.array(params_mixing_matrix_init_sample)

            if 'empirical_variance_r' in dictionnary_additional_parameters:
                print(
                    'Setting the empirical variance for r to the one provided in the additional parameters!', flush=True
                )
                initial_carry['empirical_variance_r'] = dictionnary_additional_parameters['empirical_variance_r']
            if 'empirical_variance_Bf' in dictionnary_additional_parameters:
                print(
                    'Setting the empirical variance for Bf to the one provided in the additional parameters!',
                    flush=True,
                )
                initial_carry['empirical_variance_Bf'] = dictionnary_additional_parameters['empirical_variance_Bf']
            if 'mean_r' in dictionnary_additional_parameters:
                print('Setting the mean value for r to the one provided in the additional parameters!', flush=True)
                initial_carry['mean_r'] = jnp.array(dictionnary_additional_parameters['mean_r']).squeeze()
            if 'mean_Bf' in dictionnary_additional_parameters:
                print('Setting the mean value for Bf to the one provided in the additional parameters!', flush=True)
                initial_carry['mean_Bf'] = dictionnary_additional_parameters['mean_Bf']

            assert (initial_carry['empirical_variance_r'] > 0).all()
            assert (initial_carry['empirical_variance_Bf'] > 0).all()

        ## Starting the Gibbs sampling !!!!
        time_start_sampling = time.time()
        # Start sampling !!!
        last_sample, all_samples = jlax.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        time_full_chain = (time.time() - time_start_sampling) / 60
        print(f'End of Gibbs chain in {time_full_chain} minutes, saving all files !', flush=True)

        # Saving the samples as attributes of the Sampler object
        time_start_updating = time.time()
        self.update_samples(all_samples)
        time_end_updating = (time.time() - time_start_updating) / 60
        print(f'End of updating in {time_end_updating} minutes', flush=True)

        # Saving step-sizes if SCAM is used
        if self.use_scam_step_size:
            self.all_empirical_variance_Bf = all_samples['empirical_variance_Bf']
            self.all_empirical_variance_r = all_samples['empirical_variance_r']

            # Saving the corresponding mean values for testing purposes
            self.all_mean_r = all_samples['mean_r']
            self.all_mean_Bf = all_samples['mean_Bf']

        self.number_iterations_done = self.number_iterations_sampling

        last_sample['number_iterations_done'] = self.number_iterations_done

        print('Last key PRNG', last_sample['PRNGKey'], flush=True)
        self.last_PRNGKey = last_sample['PRNGKey']

        ## Saving the last sample
        self.last_sample = last_sample
