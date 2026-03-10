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
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import checkpoint, config
from jax_tqdm import scan_tqdm

from micmac.likelihood.sampling import (
    SamplingFunctions,
    multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample,
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
    component_maps_x_redcom_covariance_cell_JAX,
    frequency_alms_x_obj_red_covariance_cell_JAX,
    get_c_ells_from_red_covariance_matrix,
    get_cell_from_map_jax,
    get_reduced_matrix_from_c_ell,
    get_reduced_matrix_from_c_ell_jax,
    get_sqrt_reduced_matrix_from_matrix_jax,
    transform_alms_shape,
)
from micmac.toolbox.utils import generate_CMB  # , generate_power_spectra_CAMB

__all__ = [
    'IcarusSampler',
]

config.update('jax_enable_x64', True)


class IcarusSampler(SamplingFunctions):
    def __init__(
        self,
        nside,
        lmax,
        nstokes,
        frequency_array,
        freq_noise_c_ell,
        pos_special_freqs=[],
        n_components=3,
        lmin=2,
        n_iter=3,
        mask=None,
        separate_CG=True,
        limit_iter_cg=200,
        limit_iter_cg_eta=200,
        tolerance_CG=1e-8,
        atol_CG=1e-8,
        save_CMB_chain_maps=False,
        save_all_Bf_params=True,
        save_s_c_spectra=False,
        sample_r_Metropolis=True,
        sample_C_inv_Wishart=False,
        sample_F=False,
        sample_F_indep=False,
        F_ell_IW_cut=2,
        save_redcom_F=False,
        simultaneous_accept_rate=True,
        non_centered_moves=False,
        save_intermediary_centered_moves=False,
        limit_r_value=False,
        below_0_min_r_value=True,
        min_r_value=None,
        use_alm_sampling_r=False,
        lmin_BB=None,
        classical_Gibbs=False,
        acceptance_posdef=False,
        use_scam_step_size=False,
        burn_in_scam=50,
        s_param_scam=(2.4) ** 2,
        epsilon_param_scam_r=1e-10,
        epsilon_param_scam_Bf=1e-11,
        scam_iteration_updates=50,
        templates=None,
        boundary_Bf=None,
        boundary_r=None,
        boundary_Sf=None,
        step_size_r=1e-4,
        covariance_Bf=None,
        covariance_Sf=None,
        indexes_free_Bf=False,
        indices_fixed_Bf=False,
        number_iterations_sampling=100,
        number_iterations_done=0,
        seed=0,
        disable_chex=False,
        instrument_name='LiteBIRD',
    ):
        """
        Main MICMAC Harmonic sampling object to initialize and launch the Metropolis-Hastings (MH) sampling in harmonic domain.
        The MH sampling will store Bf and r parameters.

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

        templates: array[int]
            Array maps with patch ids ([freq, comp, pix])
            WARNING: The spatial variability is not currently supported, but will be passed to MicmacSampler obj when using create_Harmonic_MicmacSampler_from_MicmacSampler_obj

        boundary_Bf: None or array[float] (optional)
            minimum and maximum Bf values accepted for Bf sample, set to [-inf,inf] for each Bf parameter if None, default None
        boundary_r: None or array[float] (optional)
            minimum and maximum r values accepted for r sample, set to [-inf,inf] if None, default None

        step_size_r: float (optional)
            step size for the Metropolis-Hastings sampling of r, default 1e-4
        covariance_Bf: None or array[float] of dimensions [(n_frequencies-len(pos_special_freqs))*(n_components-1), (n_frequencies-len(pos_special_freqs))*(n_components-1)] (optional)
            covariance for the Metropolis-Hastings sampling of Bf ; will be repeated if multiresoltion case, default None
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
            templates=templates,
        )

        # CMB parameters
        assert (freq_noise_c_ell.shape == (self.n_frequencies, self.n_frequencies, self.lmax + 1 - self.lmin)) or (
            freq_noise_c_ell.shape == (self.n_frequencies, self.n_frequencies, self.lmax + 1)
        )
        self.freq_noise_c_ell = freq_noise_c_ell

        # Metropolis-Hastings step-size and covariance parameters
        self.covariance_Sf = covariance_Sf
        self.covariance_Bf = covariance_Bf
        self.step_size_r = step_size_r
        self.covariance_dict = {'r': step_size_r**2, 'Bf': covariance_Bf, 'Sfgs': covariance_Sf}

        if boundary_Bf is None:
            boundary_Bf = jnp.zeros((2, (self.n_frequencies - len(self.pos_special_freqs)) * (self.n_components - 1)))
            boundary_Bf = boundary_Bf.at[0, :].set(-jnp.inf)
            boundary_Bf = boundary_Bf.at[1, :].set(jnp.inf)
        if boundary_r is None:
            # boundary_r = jnp.array([-jnp.inf, jnp.inf])
            boundary_r = jnp.array([0, jnp.inf])
        if boundary_Sf is None:
            boundary_Sf = jnp.zeros((2, self.n_components - 1, self.nstokes, (self.lmax - self.lmin + 1)))
            boundary_Sf = boundary_Sf.at[0, :].set(-jnp.inf)
            # boundary_Sf = boundary_Sf.at[0, :].set(0)
            boundary_Sf = boundary_Sf.at[1, :].set(jnp.inf)
        assert np.array(boundary_Bf).shape == (
            2,
            (self.n_frequencies - len(self.pos_special_freqs)) * (self.n_components - 1),
        )
        assert np.array(boundary_r).shape == (2,)
        assert np.array(boundary_Sf).shape == (
            2,
            self.n_components - 1,
            2,
            (self.lmax - self.lmin + 1),
        )
        self.boundary_dict = {'r': boundary_r, 'Bf': boundary_Bf, 'Sfgs': boundary_Sf}

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

        # Sampling parameters
        if indices_fixed_Bf is False:
            # If given as False, then we sample all Bf
            indices_fixed_Bf = jnp.empty(0)
            self.indices_fixed_Bf = jnp.array(indices_fixed_Bf)
        else:
            self.indices_fixed_Bf = jnp.array(indices_fixed_Bf)
            assert (
                jnp.size(self.indices_fixed_Bf) <= self.len_params
            )  # The number of free parameters should be less than the total number of parameters
            assert (
                jnp.max(self.indices_fixed_Bf) <= self.len_params
            )  # The indexes should be in the range of the total number of parameters
            assert (
                jnp.min(self.indices_fixed_Bf) >= 0
            )  # The indexes should be in the range of the total number of parameters

            self.indexes_free_Bf = jnp.delete(self.indexes_free_Bf, self.indices_fixed_Bf)
            assert not jnp.isin(self.indices_fixed_Bf, self.indexes_free_Bf).any()

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
        self.all_foreground_covariance_samples = jnp.empty(0)
        self.all_samples_r = jnp.empty(0)
        self.all_samples_F_ell = jnp.empty(0)
        self.all_samples_wiener_filter_maps = jnp.empty(0)
        self.all_samples_fluctuation_maps = jnp.empty(0)
        self.all_samples_CMB_c_ell = jnp.empty(0)
        self.all_samples_s_c_spectra = jnp.empty(0)
        self.all_samples_combined_maps = jnp.empty(0)

        self.limit_iter_cg = int(limit_iter_cg)
        self.limit_iter_cg_eta = int(limit_iter_cg_eta)
        self.tolerance_CG = float(tolerance_CG)
        self.atol_CG = float(atol_CG)
        self.separate_CG = bool(separate_CG)
        self.save_CMB_chain_maps = bool(save_CMB_chain_maps)
        self.save_all_Bf_params = bool(save_all_Bf_params)
        self.save_s_c_spectra = bool(save_s_c_spectra)
        self.sample_r_Metropolis = bool(sample_r_Metropolis)
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)
        self.sample_F = bool(sample_F)
        self.sample_F_indep = bool(sample_F_indep)
        self.F_ell_IW_cut = int(F_ell_IW_cut)
        self.save_redcom_F = bool(save_redcom_F)
        self.simultaneous_accept_rate = bool(simultaneous_accept_rate)
        self.non_centered_moves = bool(non_centered_moves)
        self.save_intermediary_centered_moves = bool(save_intermediary_centered_moves)
        self.limit_r_value = bool(limit_r_value)
        self.below_0_min_r_value = bool(below_0_min_r_value)
        self.min_r_value = min_r_value
        self.use_alm_sampling_r = bool(use_alm_sampling_r)
        self.lmin_BB = lmin_BB
        self.classical_Gibbs = bool(classical_Gibbs)
        self.acceptance_posdef = bool(acceptance_posdef)
        self.use_scam_step_size = bool(use_scam_step_size)
        self.burn_in_scam = int(burn_in_scam)
        self.s_param_scam = float(s_param_scam)
        self.epsilon_param_scam_r = float(epsilon_param_scam_r)
        self.epsilon_param_scam_Bf = float(epsilon_param_scam_Bf)
        self.scam_iteration_updates = int(scam_iteration_updates)

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

        # Define the indices to consider
        indices_polar = np.array([1, 2, 4])

        # Generate CMB from CAMB
        # theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = self.generate_CMB(return_spectra=False)
        theoretical_r0_total, theoretical_r1_tensor = generate_CMB(
            nside=self.nside, lmax=self.lmax, nstokes=self.nstokes
        )
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

    def update_samples_MH(self, all_samples):
        """
        Update the samples with new samples to add for r and Bf

        Parameters
        ----------
        all_samples: dictionary
            dictionary of all the samples to update
        """
        # Update the samples of r
        self.all_samples_r = self.update_variable(self.all_samples_r, all_samples['r'])
        # Update the samples of Bf
        self.all_params_mixing_matrix_samples = self.update_variable(
            self.all_params_mixing_matrix_samples, all_samples['Bf']
        )
        self.all_foreground_covariance_samples = self.update_variable(
            self.all_foreground_covariance_samples, all_samples['Sfgs']
        )

    def update_samples(self, all_samples):
        """
        Update the samples with new samples to add

        Parameters
        ----------
        all_samples: dictionary
            dictionary of all the samples to update
        """

        # Update the CMB chain maps if they were saved
        if self.save_CMB_chain_maps:
            if self.separate_CG:
                self.all_samples_wiener_filter_maps = self.update_variable(
                    self.all_samples_wiener_filter_maps, all_samples['wiener_filter_term']
                )
                self.all_samples_fluctuation_maps = self.update_variable(
                    self.all_samples_fluctuation_maps, all_samples['fluctuation_maps']
                )
            else:
                self.all_samples_combined_maps = self.update_variable(
                    self.all_samples_combined_maps, all_samples['combined_maps']
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

        if self.save_redcom_F:
            all_samples_F_ell = all_samples['redcom_cov_matrix_fgs_sample']
            self.all_samples_F_ell = self.update_variable(self.all_samples_F_ell, all_samples_F_ell)

        # Update the mixing matrix Bf parameters if they were sampled
        if self.save_all_Bf_params:
            self.all_params_mixing_matrix_samples = self.update_variable(
                self.all_params_mixing_matrix_samples, all_samples['params_mixing_matrix_sample']
            )

    def update_one_sample(self, one_sample):
        """
        Update the samples with one sample to add
        """

        if self.save_CMB_chain_maps:
            if self.separate_CG:
                self.all_samples_wiener_filter_maps = self.update_variable(
                    self.all_samples_wiener_filter_maps, jnp.expand_dims(one_sample['wiener_filter_term'], axis=0)
                )
                self.all_samples_fluctuation_maps = self.update_variable(
                    self.all_samples_fluctuation_maps, jnp.expand_dims(one_sample['fluctuation_maps'], axis=0)
                )
            else:
                self.all_samples_combined_maps = self.update_variable(
                    self.all_samples_combined_maps, jnp.expand_dims(one_sample['combined_maps'], axis=0)
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
                self.all_samples_r = self.update_variable(self.all_samples_r, jnp.array(one_sample['r_sample']))

        if self.sample_F:
            self.all_samples_F_ell = self.update_variable(
                self.all_samples_F_ell,
                jnp.expand_dims(one_sample['redcom_cov_matrix_fgs_sample'], axis=0),
            )

        if self.save_all_Bf_params:
            self.all_params_mixing_matrix_samples = self.update_variable(
                self.all_params_mixing_matrix_samples,
                jnp.expand_dims(one_sample['params_mixing_matrix_sample'], axis=0),
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

        JAX_input_freq_maps = jnp.array(input_freq_maps)

        def get_freq_alm(num_frequency):
            input_map_extended = jnp.vstack(
                (jnp.zeros_like(JAX_input_freq_maps[num_frequency, 0]), JAX_input_freq_maps[num_frequency, ...])
            )  ## Adding empty temperature map

            all_alms = jnp.array(
                pure_call_map2alm(input_map_extended, lmax=self.lmax)
            )  ## Getting alms for all stokes parameters

            return all_alms[3 - self.nstokes :, ...]  ## Removing the empty temperature alms

        return jax.vmap(get_freq_alm)(jnp.arange(self.n_frequencies))  ## Getting alms for all frequencies

    def perform_harmonic_MH(
        self,
        input_freq_maps,
        init_params_mixing_matrix,
        init_params_fgs_covariance,
        theoretical_r0_total,
        theoretical_r1_tensor,
        initial_guess_r=0,
        covariance_dict=None,
        input_freq_alms=None,
        print_bool=True,
    ):
        """
        Perform Metropolis Hastings to find the best r and Bf in harmonic domain.
        The chains will be stored as object attributes:
            - all_samples_r for r
            - all_params_mixing_matrix_samples for Bf

        Parameters
        ----------
        input_freq_maps : array[float] of dimensions [n_frequencies,nstokes,n_pix]
            input frequency maps
        init_params_mixing_matrix : array[float] of dimensions [n_frequencies-len(pos_special_freqs), n_components-1]
            initial parameters for the mixing matrix
        theoretical_r0_total : array[float] of dimensions [lmax+1-lmin, number_correlations, number_correlations]
            theoretical covariance matrix for the CMB scalar modes
        theoretical_r1_tensor : array[float] of dimensions [lmax+1-lmin, number_correlations, number_correlations]
            theoretical covariance matrix for the CMB tensor modes
        initial_guess_r : float (optional)
            initial guess for r, default 0
        covariance_Bf_r : None or array[float] of dimensions [(n_frequencies-len(pos_special_freqs))*(n_components-1) + 1, (n_frequencies-len(pos_special_freqs))*(n_components-1) + 1] (optional)
            covariance for the Metropolis-Hastings sampling of Bf and r, default None
        input_freq_alms : array[float] of dimensions [n_frequencies,nstokes,(lmax + 1) * (lmax // 2 + 1)] (optional)
            if provided, input_freq_alms is used instead of input_freq_maps for the MH steps
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

        ## Testing the initial mixing matrix
        if len(init_params_mixing_matrix.shape) == 1:
            assert len(init_params_mixing_matrix) == (self.n_frequencies - len(self.pos_special_freqs)) * (
                self.n_components - 1
            )
        else:
            # assert len(init_params_mixing_matrix.shape) == 2
            assert init_params_mixing_matrix.shape[0] == (self.n_frequencies - len(self.pos_special_freqs))
            assert init_params_mixing_matrix.shape[1] == (self.n_components - 1)

        ## Testing the initial fgs cov matrix
        assert init_params_fgs_covariance.shape[0] == (self.n_components - 1)
        assert init_params_fgs_covariance.shape[1] == (self.nstokes)

        # Preparing for the full Metropolis-Hatings sampling

        ## Initial guesses preparation
        params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix).ravel(order='F')

        ## Preparing the JAX PRNG key from the seed of the object
        if np.size(self.seed) == 1:
            PRNGKey = random.PRNGKey(self.seed)
        elif np.size(self.seed) == 2:
            PRNGKey = jnp.array(self.seed, dtype=jnp.uint32)
        else:
            raise ValueError('Seed should be either a scalar or a 2D array interpreted as a JAX PRNG Key!')

        ## Preparing the step-size for Metropolis-within-Gibbs of Bf sampling
        dimension_param_Bf = (self.n_frequencies - len(self.pos_special_freqs)) * (self.n_components - 1)
        dimension_param_Sf = 2 * (self.lmax - self.lmin + 1) * (self.n_components - 1)
        if covariance_dict is None:
            if self.covariance_Bf is None:
                raise ValueError('Please provide a covariance_Bf')
            assert (self.covariance_Bf).shape == (dimension_param_Bf, dimension_param_Bf)

            if self.covariance_Sf is None:
                raise ValueError('Please provide a covariance_Sf')
            assert (self.covariance_Sf).shape == (dimension_param_Sf, dimension_param_Sf)

            ## Building the full covariance of both Bf and r, without correlations between Bf and r
            covariance_dict = {'r': self.step_size_r**2, 'Bf': self.covariance_Bf, 'Sfgs': self.covariance_Sf}
        else:
            assert covariance_dict['r'].shape == (1)
            assert covariance_dict['Bf'].shape == (dimension_param_Bf, dimension_param_Bf)
            assert covariance_dict['Sfgs'].shape == (dimension_param_Sf, dimension_param_Sf)
        self.covariance_dict = covariance_dict
        if print_bool:
            print('Covariance r,Bf,Sfgs:', covariance_dict, flush=True)

        ## Getting alms from the input maps
        if input_freq_alms is None:
            input_freq_alms = self.get_alm_from_frequency_maps(input_freq_maps)
        ## Preparing the noise weighted alms
        freq_red_inverse_noise = jnp.einsum(
            'fgl,sk->fglsk', self.freq_noise_c_ell, jnp.eye(self.nstokes)
        )  ## Operator N^-1 in format [frequencies, frequencies, lmax+1-lmin, nstokes, nstokes]
        ## Applying N^-1 to the alms of the input data
        # noise_weighted_alm_data = frequency_alms_x_obj_red_covariance_cell_JAX(
        #     input_freq_alms, freq_red_inverse_noise, lmin=self.lmin
        # )
        input_freq_alms_2d = transform_alms_shape(input_freq_alms, lmax=self.lmax, transformation='healpix_to_2dlm')
        noise_weighted_alm_data_2d_cut = jnp.einsum(
            'fsLm, efLst -> etLm', input_freq_alms_2d[:, :, self.lmin : :, :], freq_red_inverse_noise
        )
        noise_weighted_alm_data_2d = jnp.zeros(jnp.shape(input_freq_alms_2d), dtype=input_freq_alms_2d.dtype)
        noise_weighted_alm_data_2d = noise_weighted_alm_data_2d.at[..., self.lmin :, :].set(
            noise_weighted_alm_data_2d_cut
        )
        indices_fixed_Bf = self.indices_fixed_Bf
        indexes_free_Bf = self.indexes_free_Bf

        print(f'Starting {self.number_iterations_sampling} iterations for harmonic run', flush=True)

        MHState = namedtuple('MHState', ['u', 'rng_key'])

        class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
            sample_field = 'u'

            def __init__(self, log_proba, covariance_dict, boundary_dict=self.boundary_dict):
                self.log_proba = log_proba
                self.covariance_dict = covariance_dict
                self.boundary_dict = boundary_dict

            def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
                return MHState(init_params, rng_key)

            def sample(self, state, model_args, model_kwargs):
                """
                One Metropolis-Hastings sampling step
                """
                new_sample, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                    state,
                    dict_covariance_matrix=self.covariance_dict,
                    log_proba=self.log_proba,
                    dict_boundary=self.boundary_dict,
                    **model_kwargs,
                )
                return MHState(new_sample, rng_key)

        class MultiStepsMetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
            sample_field = 'u'

            def __init__(self, log_proba, covariance_dict, boundary_dict=self.boundary_dict):
                self.log_proba = log_proba
                self.covariance_dict = covariance_dict
                self.boundary_dict = boundary_dict

            def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
                return MHState(init_params, rng_key)

            def sample(self, state, model_args, model_kwargs):
                """
                One Metropolis-Hastings sampling step
                """
                state_r = ({'r': state[0]['r']}, state[1])
                new_sample_r, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                    state_r,
                    dict_covariance_matrix={'r': self.covariance_dict['r']},
                    log_proba=self.log_proba,
                    dict_boundary={'r': self.boundary_dict['r']},
                    fixed_parameters_dict={'Bf': state[0]['Bf'], 'Sfgs': state[0]['Sfgs']},
                    **model_kwargs,
                )
                state_Bf = ({'Bf': state[0]['Bf']}, rng_key)
                new_sample_Bf, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                    state_Bf,
                    dict_covariance_matrix={'Bf': self.covariance_dict['Bf']},
                    log_proba=self.log_proba,
                    dict_boundary={'Bf': self.boundary_dict['Bf']},
                    fixed_parameters_dict={'r': new_sample_r['r'], 'Sfgs': state[0]['Sfgs']},
                    indices_fixed=indices_fixed_Bf,
                    fixed_key='Bf',
                    **model_kwargs,
                )
                state_Sfgs = ({'Sfgs': state[0]['Sfgs']}, rng_key)
                new_sample_Sfgs, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                    state_Sfgs,
                    dict_covariance_matrix={'Sfgs': self.covariance_dict['Sfgs']},
                    log_proba=self.log_proba,
                    dict_boundary={'Sfgs': self.boundary_dict['Sfgs']},
                    fixed_parameters_dict={'r': new_sample_r['r'], 'Bf': new_sample_Bf['Bf']},
                    **model_kwargs,
                )
                new_sample = {'r': new_sample_r['r'], 'Bf': new_sample_Bf['Bf'], 'Sfgs': new_sample_Sfgs['Sfgs']}
                return MHState(new_sample, rng_key)

        class StepsByStepsMetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
            sample_field = 'u'

            def __init__(self, log_proba, covariance_dict, boundary_dict=self.boundary_dict):
                self.log_proba = log_proba
                self.covariance_dict = covariance_dict
                self.boundary_dict = boundary_dict

            def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
                return MHState(init_params, rng_key)

            def sample(self, state, model_args, model_kwargs):
                """
                One Metropolis-Hastings sampling step
                """
                state_r = ({'r': state[0]['r']}, state[1])
                new_sample_r, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                    state_r,
                    dict_covariance_matrix={'r': self.covariance_dict['r']},
                    log_proba=self.log_proba,
                    dict_boundary={'r': self.boundary_dict['r']},
                    fixed_parameters_dict={'Bf': state[0]['Bf'], 'Sfgs': state[0]['Sfgs']},
                    **model_kwargs,
                )

                covariance_Bf = self.covariance_dict['Bf']
                boundary_Bf = self.boundary_dict['Bf']
                log_proba_fn = self.log_proba

                n_indices = len(indexes_free_Bf)
                all_fixed_indices = jnp.array([jnp.delete(jnp.arange(n_indices), i) for i in range(n_indices)])

                def scan_body(carry, inputs):
                    current_Bf, rng_key = carry
                    Bf_id, indices_fixed = inputs

                    state_Bf = ({'Bf': current_Bf}, rng_key)
                    new_sample_Bf, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                        state_Bf,
                        dict_covariance_matrix={'Bf': covariance_Bf},
                        log_proba=log_proba_fn,
                        dict_boundary={'Bf': boundary_Bf},
                        fixed_parameters_dict={'r': new_sample_r['r'], 'Sfgs': state[0]['Sfgs']},
                        indices_fixed=indices_fixed,
                        fixed_key='Bf',
                        **model_kwargs,
                    )

                    # Update carry with new sample for next iteration
                    return (new_sample_Bf['Bf'], rng_key), new_sample_Bf

                # Initialize carry with initial Bf state
                carry_init = (state[0]['Bf'], rng_key)

                # Run scan over both the IDs and the pre-computed fixed indices
                carry_final, all_samples = jax.lax.scan(scan_body, carry_init, (indexes_free_Bf, all_fixed_indices))

                # Extract final state
                final_Bf, rng_key = carry_final
                new_sample_Bf = {'Bf': final_Bf}

                state_Sfgs = ({'Sfgs': state[0]['Sfgs']}, rng_key)
                new_sample_Sfgs, rng_key = multivariate_Metropolis_Hasting_step_numpyro_bounded_dictionary_sample(
                    state_Sfgs,
                    dict_covariance_matrix={'Sfgs': self.covariance_dict['Sfgs']},
                    log_proba=self.log_proba,
                    dict_boundary={'Sfgs': self.boundary_dict['Sfgs']},
                    fixed_parameters_dict={'r': new_sample_r['r'], 'Bf': new_sample_Bf['Bf']},
                    **model_kwargs,
                )
                new_sample = {'r': new_sample_r['r'], 'Bf': new_sample_Bf['Bf'], 'Sfgs': new_sample_Sfgs['Sfgs']}
                return MHState(new_sample, rng_key)

        jitted_harmonic_maginal_proba = jax.jit(self.icarus_harmonic_marginal_probability_2d)
        mcmc_obj = numpyro.infer.mcmc.MCMC(
            # MultiStepsMetropolisHastings(
            #     log_proba=jitted_harmonic_maginal_proba, covariance_dict=covariance_dict
            # ),
            # MetropolisHastings(
            #     log_proba=jitted_harmonic_maginal_proba, covariance_dict=covariance_dict
            # ),
            StepsByStepsMetropolisHastings(log_proba=jitted_harmonic_maginal_proba, covariance_dict=covariance_dict),
            num_warmup=0,
            num_samples=self.number_iterations_sampling - self.number_iterations_done,
            progress_bar=True,
        )

        # Initializing r and Bf samples
        init_params_dict = {
            'r': initial_guess_r,
            'Bf': params_mixing_matrix_init_sample,
            'Sfgs': init_params_fgs_covariance,
        }

        time_start_sampling = time.time()
        ## Starting the MH sampling !!!
        mcmc_obj.run(
            PRNGKey,
            init_params=init_params_dict,
            noise_weighted_alm_data_2d=noise_weighted_alm_data_2d,
            theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor,
            theoretical_red_cov_r0_total=theoretical_red_cov_r0_total,
        )

        time_full_chain = (time.time() - time_start_sampling) / 60
        print(f'End of MH iterations for harmonic run in {time_full_chain} minutes !', flush=True)

        posterior_samples = mcmc_obj.get_samples()
        if print_bool:
            print('Summary of the run', flush=True)
            mcmc_obj.print_summary()

        # Saving the samples as attributes of the Sampler object
        self.update_samples_MH(posterior_samples)
        self.last_sample = {
            'r_sample': posterior_samples['r'][-1],
            'params_mixing_matrix_sample': posterior_samples['Bf'][-1],
            'params_foreground_covariance_sample': posterior_samples['Sfgs'][-1],
            'input_freq_alms': input_freq_alms,
        }
        self.number_iterations_done = self.number_iterations_sampling
        self.last_PRNGKey = PRNGKey

    def perform_Gibbs_sampling(
        self,
        input_freq_maps,
        CMB_c_ell,
        init_params_mixing_matrix,
        init_params_fgs_covariance,  # TODO: asserts and put in reduced form
        initial_guess_r=1e-8,
        initial_wiener_filter_term=None,
        initial_fluctuation_maps=None,
        initial_combined_maps=None,
        theoretical_r0_total=None,
        theoretical_r1_tensor=None,
        **dictionnary_additional_parameters,
    ):
        r"""
        Perform sampling steps with:
            1. A CG for the Wiener filter (WF) and fluctuation variables s: (s - s_{WF})^t (S^{-1} + N_c^{-1}) (s - s_{WF})
            3. The c_ell sampling, either by parametrizing it by r or by sampling an inverse Wishart distribution
            3. The F_ell sampling, by sampling an inverse Wishart distribution
            4. Mixing matrix Bf sampling with: (d - B_c s_c - B_f s_f)^t N^{-1} (d - B_c s_c - B_f s_f)

        The results of the chain will be stored in the class attributes, depending if the save options are put to True or False:
            - self.all_samples_wiener_filter_maps (if self.save_CMB_chain_maps and self.separate_CG is True)
            - self.all_samples_fluctuation_maps (if self.save_CMB_chain_maps and self.separate_CG is True)
            - self.all_samples_combined_maps (if self.save_CMB_chain_maps is True and self.separate_CG is False)
            - self.all_samples_r (if self.sample_r_Metropolis is True)
            - self.all_samples_CMB_c_ell (if self.sample_C_inv_Wishart is True)
            - self.all_samples_F_ell (if self.sample_F is True)
            - self.all_params_mixing_matrix_samples (always)

        This same function can be used to continue a chain from a previous run, by giving the number of iterations already done in the IcarusSampler object,
        giving the chains to the attributes of the object, and giving the last iteration results as initial guesses.

        Parameters
        ----------
        input_freq_maps: array[float] of dimensions [frequencies, nstokes, n_pix]
            input frequency maps
        CMB_c_ell: array[float] of dimensions [number_correlations, lmax+1]
            CMB power spectra, where number_correlations is the number of auto- and cross-correlations relevant considering the number of Stokes parameters
        init_params_mixing_matrix: array[float] of dimensions [len_params]
            initial parameters for the mixing matrix elements Bf; expected to be given flattened as [Bf_s1, Bf_s2, ..., Bf_sn, Bf_d1, ..., Bf_dn]
        init_params_fgs_covariance: array[float] of dimensions [lmax - lmin + 1, n_comps - 1, nstokes, n_comps - 1, nstokes]
            initial parameters for the foregrounds covariance; expected to be given in a component stokes form
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

        if self.separate_CG:
            ## Testing the initial WF term, or initialize it properly
            if initial_wiener_filter_term is None:
                wiener_filter_term = jnp.zeros((self.n_components, self.nstokes, self.n_pix))
            else:
                assert len(initial_wiener_filter_term.shape) == 3
                assert initial_wiener_filter_term.shape == (self.n_components, self.nstokes, self.n_pix)
                wiener_filter_term = initial_wiener_filter_term

            ## Testing the initial fluctuation term, or initialize it properly
            if initial_fluctuation_maps is None:
                fluctuation_maps = jnp.zeros((self.n_components, self.nstokes, self.n_pix))
            else:
                assert len(initial_fluctuation_maps.shape) == 3
                assert initial_fluctuation_maps.shape == (self.n_components, self.nstokes, self.n_pix)
                fluctuation_maps = initial_fluctuation_maps
        else:
            if (
                initial_wiener_filter_term is None
                and initial_fluctuation_maps is None
                and initial_combined_maps is None
            ):
                combined_maps = jnp.zeros((self.n_components, self.nstokes, self.n_pix))
            elif initial_combined_maps is not None:
                assert len(initial_combined_maps.shape) == 3
                assert initial_combined_maps.shape == (self.n_components, self.nstokes, self.n_pix)
                combined_maps = initial_combined_maps
            else:
                initial_combined_maps = initial_wiener_filter_term + initial_fluctuation_maps
                assert len(initial_combined_maps.shape) == 3
                assert initial_combined_maps.shape == (self.n_components, self.nstokes, self.n_pix)
                combined_maps = initial_combined_maps

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

        ## Testing the initial CMB spectra spectra given
        if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):
            CMB_c_ell = CMB_c_ell[
                indices_to_consider, :
            ]  # Selecting only the relevant auto- and cross-correlations for polarization

        assert len(CMB_c_ell.shape) == 2
        assert CMB_c_ell.shape[1] == self.lmax + 1

        if self.sample_F:
            assert self.F_ell_IW_cut >= self.lmin

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
        print(self.below_0_min_r_value)
        print(self.min_r_value)
        if self.below_0_min_r_value and (self.min_r_value is None):
            print('Setting min_r_value so that C(r) is positive definite', flush=True)
            self.min_r_value = -np.min(theoretical_red_cov_r0_total[:, 1, 1] / theoretical_red_cov_r1_tensor[:, 1, 1])
        assert (
            initial_guess_r > self.min_r_value
        ), f'Not allowing first guess for r {initial_guess_r} to have value lower than min_r_value {self.min_r_value}'

        # Preparing for the full Gibbs sampling
        len_pos_special_freqs = len(self.pos_special_freqs)

        assert (
            np.shape(init_params_fgs_covariance)[0] == self.lmax - self.lmin + 1
        ), 'Incorrect dimension for fgs covariance'
        assert (
            np.shape(init_params_fgs_covariance)[1] == self.n_components - 1
        ), 'Incorrect components dimension for fgs covariance'
        assert np.shape(init_params_fgs_covariance)[2] == self.nstokes, 'Incorrect stokes dimension for fgs covariance'

        ## Initial guesses preparation
        ## CMB covariance preparation in the format [lmax,nstokes,nstokes]
        red_cov_matrix = get_reduced_matrix_from_c_ell(CMB_c_ell)[self.lmin :, ...]

        ## parameters of the mixing matrix
        params_mixing_matrix_init_sample = jnp.array(init_params_mixing_matrix, copy=True)

        # Preparing the sampling functions
        if self.separate_CG:
            ## Function to compute the Wiener filter term
            sampling_func_WF = self.solve_generalized_wiener_filter_term_multi_components
            ## Function to sample the fluctuation maps
            sampling_func_Fluct = self.get_fluctuating_term_maps_multi_components
        else:
            sampling_func_combined = self.solve_combined_maps_multi_components_lineax
        ## Function to sample the CMB covariance from inverse Wishart
        func_get_inverse_wishart_sampling_from_c_ells = self.get_inverse_wishart_sampling_from_c_ells
        ## Function to sample the CMB covariance parametrize from r
        r_sampling_MH = single_Metropolis_Hasting_step
        # r_sampling_MH = bounded_single_Metropolis_Hasting_step
        if self.sample_r_Metropolis:
            log_proba_r = self.get_conditional_proba_C_from_r_wBB

        ## Function to sample the mixing matrix free parameters through the difference of the log-proba, to have only one CG done
        jitted_Bf_func_sampling = jax.jit(
            self.get_conditional_proba_mixing_matrix_v3_pixel_icarus_JAX,
        )
        sampling_func = separate_single_MH_step_index_v2b

        if self.simultaneous_accept_rate:
            ## More efficient version of the mixing matrix sampling

            ## MH step function to sample the mixing matrix free parameters with patches simultaneous computed accept rate
            print('Using simultaneous accept rate version of mixing matrix sampling !!!', flush=True)
            # print(
            #     '---- ATTENTION: This assumes all patches are distributed in the same way for all parameters !',
            #     flush=True,
            # )
            jitted_Bf_func_sampling = jax.jit(
                self.get_conditional_proba_mixing_matrix_v3_pixel_icarus_JAX,
            )
            # sampling_func = separate_single_MH_step_index_v4_pixel  # separate_single_MH_step_index_v4b_pixel
            # if (self.n_patches != self.n_patches[0]).any():
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
                    carry | ((index_Bf >= indexes_patches_Bf) & (index_Bf < indexes_patches_Bf + self.n_patches)),
                    index_Bf,
                )

            condition, _ = jlax.scan(which_interval, jnp.zeros_like(self.n_patches, dtype=bool), self.indexes_free_Bf)

            first_indices_patches_free_Bf = indexes_patches_Bf[condition]
            max_len_patches_Bf = int(np.max(self.n_patches[condition]))
            n_patches = self.n_patches[condition]

            print('First indices patches free Bf', first_indices_patches_free_Bf, flush=True)
            print('Max length patches Bf', max_len_patches_Bf, flush=True)
            print(
                'Number of patches to consider for the simultaneous accept rate Bf sampling',
                n_patches,
                flush=True,
            )

            indices_templates_in_params_long = jnp.zeros_like(self.indexes_free_Bf)
            for i, index in enumerate(self.indexes_free_Bf):
                indices_templates_in_params_long = indices_templates_in_params_long.at[i].set(
                    jnp.where(index >= first_indices_patches_free_Bf)[0][-1]
                )
            print('Indices templates in params long', indices_templates_in_params_long, flush=True)

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
                if self.covariance_Bf.shape[0] != (self.n_components - 1) * (
                    self.n_frequencies - len_pos_special_freqs
                ):
                    raise ValueError(
                        f'Covariance matrix for Bf is not of the right shape with shape {self.covariance_Bf.shape[0]}, it cannot be properly expanded with the considered multipatch distribution!'
                    )

                if (
                    self.n_patches is not None and (self.n_patches == self.n_patches[0]).all()
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
                    extended_array[1:-1] = self.indexes_b.ravel(order='F')[1:]
                    extended_array[-1] = self.len_params

                    for i in range(
                        self.n_patches.size
                    ):  # Loop over the patches to update the step-size for each patch size
                        initial_step_size_Bf = initial_step_size_Bf.at[extended_array[i] : extended_array[i + 1]].set(
                            previous_initial_Bf[i]
                        )
                    initial_step_size_Bf = initial_step_size_Bf.at[extended_array[-1] :].set(previous_initial_Bf[-1])

                print('New step-size Bf', initial_step_size_Bf, flush=True)

        ## Few prints to re-check the toml parameters chosen
        if self.sample_r_Metropolis:
            print('Sample for r instead of C!', flush=True)
            if self.limit_r_value:
                print(f'Limiting the r value to be superior to {self.min_r_value} !', flush=True)

        # Few steps to improve the speed of the code

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

        @scan_tqdm(actual_number_of_iterations, print_rate=1)
        def all_sampling_steps(carry, iteration):
            """
            1-step Gibbs sampling function, performing the following:
            - Sampling of s, for the constrained CMB and foregrounds map realization (s should be of dimension ???); sampling both Wiener filter and fluctuation maps
            - Sampling of C or r parametrizing C, for the CMB covariance matrix
            - Sampling of F, for the foregrounds covariance matrix
            - Sampling of the free Af parameters, for the mixing matrix

            Parameters
            ----------
            carry: dictionary
                dictionary containing the following variables at 1 iteration depending on the option chosen: WF maps, fluctuation maps, component covariance, r samples, Bf samples, PRNGKey
            iteration: int
                current iteration number

            Returns
            -------
            new_carry: dictionary
                dictionary containing the following variables at the next iteration: WF maps, fluctuation maps, component covariance, r sample, Bf sample, PRNGKey
            all_samples: dictionary
                dictionary containing the variables to save as chains, so depending on the options chosen: eta maps, WF maps, fluctuation maps, component covariance, r sample, Bf sample
            """

            # Extracting the JAX PRNG key from the carry
            PRNGKey = carry['PRNGKey']

            # Preparing the new carry and all_samples to save the chains
            new_carry = dict()
            all_samples = dict()

            # Preparing a new PRNGKey for sampling
            PRNGKey, subPRNGKey = random.split(PRNGKey)

            # Extracting the mixing matrix parameters and initializing the new one
            mixing_matrix_sampled = self.get_B_from_params(carry['params_mixing_matrix_sample'], jax_use=True)

            # Few checks for the mixing matrix
            chx.assert_shape(mixing_matrix_sampled, (self.n_frequencies, self.n_components, self.n_pix))

            # Application of new mixing matrix to the noise covariance and extracted CMB map from data
            invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, mixing_matrix_sampled, jax_use=True)
            BtinvN_sqrt = get_BtinvN(jnp.sqrt(self.freq_inverse_noise), mixing_matrix_sampled, jax_use=True)
            s_GLS = get_Wd(self.freq_inverse_noise, mixing_matrix_sampled, input_freq_maps, jax_use=True)

            # Sampling step 1: sampling of Gaussian variable s, contrained maps realization

            ## Building full covariance
            redcom_cov_matrix_sqrt = jnp.zeros(
                shape=(self.lmax - self.lmin + 1, self.n_components, self.nstokes, self.n_components, self.nstokes)
            )

            cov_matrix_cmb_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(carry['red_cov_matrix_sample'])
            ## Geting the square root matrix of the sampled total covariance (assuming dimension , n_multipole, ncomps, nstokes, ncomps, nstokes) #TODO: make sure that the input redcom_cov_matrix_sample is of the right dimension
            concat_cov_matrix_fgs_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(
                carry['redcom_cov_matrix_fgs_sample'].reshape(
                    (
                        self.lmax - self.lmin + 1,
                        (self.n_components - 1) * self.nstokes,
                        (self.n_components - 1) * self.nstokes,
                    )
                )
            )

            redcom_cov_matrix_sqrt = redcom_cov_matrix_sqrt.at[:, 0, :, 0, :].set(cov_matrix_cmb_sqrt)
            redcom_cov_matrix_sqrt = redcom_cov_matrix_sqrt.at[:, 1:, :, 1:, :].set(
                concat_cov_matrix_fgs_sqrt.reshape(
                    (
                        self.lmax - self.lmin + 1,
                        self.n_components - 1,
                        self.nstokes,
                        self.n_components - 1,
                        self.nstokes,
                    )
                )
            )

            concat_cov_matrix_sqrt = redcom_cov_matrix_sqrt.reshape(
                self.lmax - self.lmin + 1, self.n_components * self.nstokes, self.n_components * self.nstokes
            )

            # Preparing the preconditioner to use for the sampling of the maps
            precond_func_s = None
            if use_precond:
                N_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled.mean(axis=2))
                redcom_N_ell = jnp.einsum('cdl,sk->lcsdk', N_ell, np.eye(self.nstokes))
                inv_redcom_N_ell = jnp.linalg.pinv(
                    redcom_N_ell.reshape(
                        self.lmax - self.lmin + 1, self.n_components * self.nstokes, self.n_components * self.nstokes
                    )
                ).reshape(self.lmax - self.lmin + 1, self.n_components, self.nstokes, self.n_components, self.nstokes)
                redcom_preconditioner_s = jnp.linalg.pinv(
                    jnp.eye(self.n_components * self.nstokes)
                    + jnp.einsum(
                        'labcd,lcdef,lefgh->labgh', redcom_cov_matrix_sqrt, inv_redcom_N_ell, redcom_cov_matrix_sqrt
                    ).reshape(
                        self.lmax - self.lmin + 1, self.n_components * self.nstokes, self.n_components * self.nstokes
                    )  # /  f_sky ** 2
                ).reshape(
                    self.lmax - self.lmin + 1, self.n_components, self.nstokes, self.n_components, self.nstokes
                )  # *  f_sky

                precond_func_s = lambda x: component_maps_x_redcom_covariance_cell_JAX(
                    x.reshape((self.n_components, self.nstokes, self.n_pix)),
                    redcom_preconditioner_s,
                    nside=self.nside,
                    lmin=self.lmin,
                    n_iter=self.n_iter,
                ).ravel()
            if self.separate_CG:
                ## Computing an initial guess closer to the actual start of the CG for the Wiener filter
                initial_guess_WF = component_maps_x_redcom_covariance_cell_JAX(
                    carry['wiener_filter_term'],
                    jnp.linalg.pinv(concat_cov_matrix_sqrt).reshape(
                        (self.lmax - self.lmin + 1, self.n_components, self.nstokes, self.n_components, self.nstokes)
                    ),
                    nside=self.nside,
                    lmin=self.lmin,
                    n_iter=self.n_iter,
                )
                ## Sampling the Wiener filter term #TODO: change sampling function to the one with appropriate dimensionality
                # redcom_N = jnp.einsum('cdp,sk->pcsdk', invBtinvNB * hp.nside2resol(self.nside) ** 2, jnp.eye(self.nstokes))
                redcom_N = jnp.einsum('cdp,sk->pcsdk', invBtinvNB, jnp.eye(self.nstokes))
                redcom_N_inv = jnp.copy(redcom_N)
                redcom_N_inv = redcom_N_inv.at[self.mask != 0, ...].set(
                    jnp.linalg.pinv(
                        redcom_N.reshape(self.n_pix, self.n_components * self.nstokes, self.n_components * self.nstokes)
                    ).reshape(self.n_pix, self.n_components, self.nstokes, self.n_components, self.nstokes)
                )

                new_carry['wiener_filter_term'] = sampling_func_WF(
                    s_GLS,
                    redcom_cov_matrix_sqrt,
                    redcom_N_inv,
                    initial_guess=initial_guess_WF,
                    precond_func=precond_func_s,
                )

                ## Preparing the random variables for the fluctuation term
                PRNGKey, new_subPRNGKey = random.split(PRNGKey)
                map_random_realization_xi = None
                map_random_realization_chi = None

                ## Getting the fluctuation maps terms, for the variance of the variable s
                initial_guess_Fluct = component_maps_x_redcom_covariance_cell_JAX(
                    carry['fluctuation_maps'],
                    jnp.linalg.pinv(concat_cov_matrix_sqrt).reshape(
                        (self.lmax - self.lmin + 1, self.n_components, self.nstokes, self.n_components, self.nstokes)
                    ),
                    nside=self.nside,
                    lmin=self.lmin,
                    n_iter=self.n_iter,
                )
                ## Sampling the fluctuation maps #TODO: change sampling function to the one with appropriate dimensionality
                new_carry['fluctuation_maps'] = sampling_func_Fluct(
                    redcom_cov_matrix_sqrt,
                    BtinvN_sqrt,
                    redcom_N_inv,
                    new_subPRNGKey,
                    map_random_realization_xi=map_random_realization_xi,
                    map_random_realization_chi=map_random_realization_chi,
                    initial_guess=initial_guess_Fluct,
                    precond_func=precond_func_s,
                )

                ## Constructing the sampled maps: Should be of shape n_comps, n_stokes, n_multipoles
                s_sample = new_carry['fluctuation_maps'] + new_carry['wiener_filter_term']
            else:
                ## Preparing the random variables for the fluctuation term
                PRNGKey, new_subPRNGKey = random.split(PRNGKey)
                map_random_realization_xi = None
                map_random_realization_chi = None

                initial_guess_combined = component_maps_x_redcom_covariance_cell_JAX(
                    carry['combined_maps'],
                    jnp.linalg.pinv(concat_cov_matrix_sqrt).reshape(
                        (self.lmax - self.lmin + 1, self.n_components, self.nstokes, self.n_components, self.nstokes)
                    ),
                    nside=self.nside,
                    lmin=self.lmin,
                    n_iter=self.n_iter,
                )
                ## Sampling the Wiener filter term #TODO: change sampling function to the one with appropriate dimensionality
                # redcom_N = jnp.einsum('cdp,sk->pcsdk', invBtinvNB * hp.nside2resol(self.nside) ** 2, jnp.eye(self.nstokes))
                redcom_N = jnp.einsum('cdp,sk->pcsdk', invBtinvNB, jnp.eye(self.nstokes))
                redcom_N_inv = jnp.copy(redcom_N)
                redcom_N_inv = redcom_N_inv.at[self.mask != 0, ...].set(
                    jnp.linalg.pinv(
                        redcom_N.reshape(self.n_pix, self.n_components * self.nstokes, self.n_components * self.nstokes)
                    ).reshape(self.n_pix, self.n_components, self.nstokes, self.n_components, self.nstokes)
                )

                new_carry['combined_maps'] = sampling_func_combined(
                    s_GLS,
                    redcom_cov_matrix_sqrt,
                    BtinvN_sqrt,
                    redcom_N_inv,
                    new_subPRNGKey,
                    map_random_realization_xi=map_random_realization_xi,
                    map_random_realization_chi=map_random_realization_chi,
                    initial_guess=initial_guess_combined,
                    precond_func=precond_func_s,
                )

                ## Constructing the sampled maps: Should be of shape n_comps, n_stokes, n_multipoles
                s_sample = new_carry['combined_maps']
            ## Retrieving the CMB component
            s_c_sample = s_sample[0]
            s_f_sample = s_sample[1::]

            if self.save_CMB_chain_maps:
                ## Saving the sampled Wiener filter term and fluctuation maps if chosen to
                if self.separate_CG:
                    all_samples['wiener_filter_term'] = new_carry['wiener_filter_term']
                    all_samples['fluctuation_maps'] = new_carry['fluctuation_maps']
                    ## Checking the shape of the resulting maps
                    chx.assert_shape(new_carry['wiener_filter_term'], (self.n_components, self.nstokes, self.n_pix))
                    chx.assert_shape(new_carry['fluctuation_maps'], (self.n_components, self.nstokes, self.n_pix))
                    chx.assert_shape(s_sample, (self.n_components, self.nstokes, self.n_pix))
                else:
                    all_samples['combined_maps'] = new_carry['combined_maps']
                    ## Checking the shape of the resulting maps
                    chx.assert_shape(new_carry['combined_maps'], (self.n_components, self.nstokes, self.n_pix))
                    chx.assert_shape(s_sample, (self.n_components, self.nstokes, self.n_pix))

            # Sampling step 2: sampling of CMB covariance C (either r only or the entire spectrum) #TODO: modify this section

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
                new_carry['red_cov_matrix_sample'] = (
                    new_carry['red_cov_matrix_sample']
                    .at[:, 0, :, 0, :]
                    .set(
                        func_get_inverse_wishart_sampling_from_c_ells(
                            c_ells_Wishart_,
                            PRNGKey=new_subPRNGKey_2,
                            old_sample=carry['red_cov_matrix_sample'],
                            acceptance_posdef=self.acceptance_posdef,
                        )
                    )
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

                dictionary_arguments_sampling_r['lmin_BB'] = self.lmin_BB
                dictionary_arguments_sampling_r['red_sigma_ell'] = red_c_ells_Wishart_modified

                if self.lmin_BB is not None:
                    dictionary_arguments_sampling_r['theoretical_red_cov_r1_tensor'] = theoretical_red_cov_r1_tensor[
                        self.lmin_BB - self.lmin :, ...
                    ]
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

                ## Saving the r sample
                all_samples['r_sample'] = new_carry['r_sample']
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')

            # ## Checking the shape of the resulting covariance matrix, and correcting it if needed
            # if new_carry['redcom_cov_matrix_sample'].shape[0] == self.lmax + 1:
            #     new_carry['redcom_cov_matrix_sample'] = new_carry['redcom_cov_matrix_sample'][self.lmin :]

            # ## Small check on the shape of the resulting covariance matrix
            # chx.assert_shape(
            #     new_carry['redcom_cov_matrix_sample'], (self.lmax + 1 - self.lmin, self.n_components, self.nstokes, self.n_components, self.nstokes)
            # )

            # Sampling step 3: sampling of foreground covariance Sf #TODO: input parameter ell inv Wishart
            new_redcom_F_samples = carry['redcom_cov_matrix_fgs_sample']
            if self.sample_F:
                if self.sample_F_indep:
                    PRNGKey, new_subPRNGKey_2 = random.split(PRNGKey)
                    keys = random.split(new_subPRNGKey_2, self.n_components - 1)
                    component_indices = jnp.arange(self.n_components - 1)

                    # Store references to avoid passing large arrays repeatedly
                    s_f_full = s_f_sample
                    init_cov_full = carry['redcom_cov_matrix_fgs_sample']

                    def matrix_to_flat(mat):
                        """Extract [EE, BB, EB] from (ncomp, nstokes, ncomp, nstokes) symmetric matrix for a single component pair (i,i)."""

                        # mat shape: (ncomp, nstokes, ncomp, nstokes)
                        # Returns flat [EE, BB, EB] for each component i
                        def extract_comp(i):
                            EE = mat[i, 0, i, 0]
                            BB = mat[i, 1, i, 1]
                            EB = mat[i, 1, i, 0]  # = BE by symmetry
                            return jnp.array([EE, BB, EB])

                        return jax.vmap(extract_comp)(jnp.arange(self.n_components - 1))

                    def flat_to_matrix(flat_comp):
                        """Reconstruct (ncomp, nstokes, ncomp, nstokes) from flat [EE, BB, EB] per component."""
                        # flat_comp shape: (ncomp, 3)
                        ncomp = flat_comp.shape[0]
                        mat = jnp.zeros((ncomp, self.nstokes, ncomp, self.nstokes))

                        def set_comp(mat, args):
                            i, flat = args
                            mat = mat.at[i, 0, i, 0].set(flat[0])  # EE
                            mat = mat.at[i, 1, i, 1].set(flat[1])  # BB
                            mat = mat.at[i, 1, i, 0].set(flat[2])  # EB
                            mat = mat.at[i, 0, i, 1].set(flat[2])  # BE = EB
                            return mat, None

                        mat, _ = jax.lax.scan(set_comp, mat, (jnp.arange(ncomp), flat_comp))
                        return mat

                    def process_single_component(component_idx, key):
                        # Access large arrays from outer scope
                        # Get C_ells for single fgs component
                        c_ells_Wishart_fgs = get_cell_from_map_jax(
                            s_f_full[component_idx], lmax=self.lmax, n_iter=self.n_iter
                        )[:, self.lmin :]

                        # Get covariance single component
                        old_cov = init_cov_full[:, component_idx, :, component_idx, :]

                        # Get F sample single component
                        new_redcom_F_sample = func_get_inverse_wishart_sampling_from_c_ells(
                            c_ells_Wishart_fgs,
                            PRNGKey=key,
                            old_sample=old_cov,
                            acceptance_posdef=self.acceptance_posdef,
                        )

                        return new_redcom_F_sample, c_ells_Wishart_fgs

                    # Use vmap to process all components in parallel
                    fgs_samples, c_ells_fgs_all = jax.vmap(process_single_component)(component_indices, keys)

                    # Update the full covariance matrix efficiently
                    def update_diagonal(i, cov):
                        return cov.at[:, i, :, i, :].set(fgs_samples[i])

                    new_redcom_F_samples = jax.lax.fori_loop(
                        0, self.n_components - 1, update_diagonal, new_redcom_F_samples
                    )

                    redcom_cov_c_ells_fgs = flat_to_matrix(c_ells_fgs_all[..., self.F_ell_IW_cut - self.lmin])
                    # Introduce sampling for this bin later.

                    def step_single_bin(prng_key, bin_mat, ell, red_sigma_ell, step_sizes):
                        """Take a single MH step for one bin's (ncomp, nstokes, ncomp, nstokes) matrix."""
                        flat = matrix_to_flat(bin_mat)  # (ncomp, 3)

                        def step_single_comp(args):
                            key, flat_i, step_i, i = args
                            prngkey, key_proposal, key_accept = random.split(key, 3)
                            key_EE, key_BB, key_EB = random.split(key_proposal, 3)

                            # Mixed proposal: LogNormal for EE, BB and Normal for EB
                            proposed_EE = dist.LogNormal(jnp.log(flat_i[0]), step_i[0]).sample(key_EE)
                            proposed_BB = dist.LogNormal(jnp.log(flat_i[1]), step_i[1]).sample(key_BB)
                            proposed_EB = dist.Normal(flat_i[2], step_i[2]).sample(key_EB)

                            sample_proposal = jnp.array([proposed_EE, proposed_BB, proposed_EB])

                            # Full vector log probability
                            accept_prob = -(
                                self.get_conditional_proba_F_ell_indep(
                                    flat_i, ell=ell, red_sigma_ell=red_sigma_ell[i, :, i, :]
                                )
                                - self.get_conditional_proba_F_ell_indep(
                                    sample_proposal, ell=ell, red_sigma_ell=red_sigma_ell[i, :, i, :]
                                )
                            )

                            new_flat_i = jnp.where(
                                jnp.log(dist.Uniform().sample(key_accept)) < accept_prob, sample_proposal, flat_i
                            )

                            # Only need determinant check now, EE and BB are guaranteed positive
                            EE, BB, EB = new_flat_i[0], new_flat_i[1], new_flat_i[2]
                            is_pos_def = EE * BB > EB**2
                            # cond_num = condition_number_2x2(EE, BB, EB)
                            # jax.debug.print("condition number {a}", a = cond_num)

                            ref_flat = matrix_to_flat(red_sigma_ell)[i]
                            scale_factor = np.array([jnp.inf, jnp.inf, jnp.inf])
                            # is_well_conditioned = cond_num < 1e6  # e.g. 1e6 or 1e8
                            is_within_bounds = jnp.all(jnp.abs(new_flat_i) < scale_factor * jnp.abs(ref_flat))

                            is_valid = is_pos_def & is_within_bounds  # & is_well_conditioned
                            return jnp.where(is_valid, new_flat_i, flat_i)

                        ncomp = flat.shape[0]
                        keys = random.split(prng_key, ncomp)
                        new_flat = jax.vmap(step_single_comp)((keys, flat, step_sizes, jnp.arange(ncomp)))  # (ncomp, 3)
                        return flat_to_matrix(new_flat)  # (ncomp, nstokes, ncomp, nstokes)

                    PRNGKey, new_subPRNGKey_2 = random.split(PRNGKey)
                    # step_sizes_F_ell_first_bin =  np.array([[0.5,0.5,20]])
                    step_sizes_F_ell_first_bin = matrix_to_flat(self.covariance_Sf[..., self.F_ell_IW_cut - self.lmin])
                    # step_sizes_F_ell_first_bin =  np.array([[0.5,0.5,20],[0.5,0.5,100]]) #np.array([[119.90482425774194,1.7337962802483704,0],[689.1487354935336,12.130559481929641,0]])
                    current_sample_F_ell_first_bin = carry['redcom_cov_matrix_fgs_sample'][
                        self.F_ell_IW_cut - self.lmin
                    ]
                    new_sample_F_ell_first_bin = step_single_bin(
                        new_subPRNGKey_2,
                        current_sample_F_ell_first_bin,
                        ell=self.F_ell_IW_cut,
                        red_sigma_ell=redcom_cov_c_ells_fgs,
                        step_sizes=step_sizes_F_ell_first_bin,
                    )
                    new_redcom_F_samples = new_redcom_F_samples.at[self.F_ell_IW_cut - self.lmin].set(
                        new_sample_F_ell_first_bin
                    )
                    # new_redcom_F_samples = new_redcom_F_samples.at[self.F_ell_IW_cut - self.lmin].set(carry['redcom_cov_matrix_fgs_sample'][self.F_ell_IW_cut - self.lmin])
                    cond_F_samples = jnp.linalg.cond(
                        new_redcom_F_samples.reshape(
                            (
                                self.lmax - self.lmin + 1,
                                (self.n_components - 1) * self.nstokes,
                                (self.n_components - 1) * self.nstokes,
                            )
                        )
                    )
                    cond_mask = (cond_F_samples > 1e7)[:, None, None, None, None]
                    new_redcom_F_samples = jnp.where(
                        cond_mask, carry['redcom_cov_matrix_fgs_sample'], new_redcom_F_samples
                    )
                else:
                    # Sample full matrix F
                    new_carry['redcom_cov_matrix_fgs_sample'] = func_get_inverse_wishart_sampling_from_c_ells(
                        c_ells_Wishart_,
                        PRNGKey=new_subPRNGKey_2,
                        old_sample=carry['redcom_cov_matrix_fgs_sample'],
                        acceptance_posdef=self.acceptance_posdef,
                    )
            new_carry['redcom_cov_matrix_fgs_sample'] = new_redcom_F_samples
            if self.save_redcom_F:
                all_samples['redcom_cov_matrix_fgs_sample'] = new_carry['redcom_cov_matrix_fgs_sample']

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
            ## Preparing the parameters to provide for the sampling of Bf

            dict_parameters_sampling_Bf = {
                'indexes_Bf': self.indexes_free_Bf,
                'full_data_without_CMB': full_data_without_CMB,
                'foreground_maps_sample': s_f_sample,
            }

            if self.simultaneous_accept_rate:
                ## Provide as well the indexes of the patches in case of the uncorrelated patches version
                dict_parameters_sampling_Bf['n_patches'] = n_patches
                dict_parameters_sampling_Bf['max_len_patches_Bf'] = max_len_patches_Bf
                dict_parameters_sampling_Bf['indexes_patches_Bf'] = first_indices_patches_free_Bf
                dict_parameters_sampling_Bf['len_indexes_Bf'] = self.len_params
                dict_parameters_sampling_Bf['indices_templates_in_params_long'] = indices_templates_in_params_long
                # TODO: Accelerate by removing indexes of indexes_patches_Bf if the corresponding patches are not in indexes_free_Bf, nor in the mask
            ## Sampling Bf !
            new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'] = sampling_func(
                random_PRNGKey=new_subPRNGKey_3,
                old_sample=carry['params_mixing_matrix_sample'],
                step_size=step_size_Bf,
                log_proba=jitted_Bf_func_sampling,
                **dict_parameters_sampling_Bf,
            )

            # Checking the shape of the resulting mixing matrix
            chx.assert_shape(new_carry['params_mixing_matrix_sample'], (self.len_params,))

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
        if self.separate_CG:
            initial_carry = {
                'wiener_filter_term': wiener_filter_term,
                'fluctuation_maps': fluctuation_maps,
                'red_cov_matrix_sample': red_cov_matrix,
                'redcom_cov_matrix_fgs_sample': init_params_fgs_covariance,
                'params_mixing_matrix_sample': params_mixing_matrix_init_sample,
                'PRNGKey': PRNGKey,
            }
        else:
            initial_carry = {
                'combined_maps': combined_maps,
                'red_cov_matrix_sample': red_cov_matrix,
                'redcom_cov_matrix_fgs_sample': init_params_fgs_covariance,
                'params_mixing_matrix_sample': params_mixing_matrix_init_sample,
                'PRNGKey': PRNGKey,
            }

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
