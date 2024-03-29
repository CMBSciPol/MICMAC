import os, sys, time
import toml
import numpy as np
import healpy as hp
import scipy
import fgbuster
import jax
import jax.lax as jlx
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
from jax_tqdm import scan_tqdm, loop_tqdm
import chex as chx
import numpyro

from .tools import *
from .jax_tools import *
from .mixingmatrix import *
from .templates_spv import *
from .noisecovar import *
from .jax_Sampling_toolbox import *
from .utils import *

from jax import config
config.update("jax_enable_x64", True)

class MICMAC_Sampler(Sampling_functions):
    def __init__(self, 
                 nside, 
                 lmax, 
                 nstokes, 
                 frequency_array, 
                 freq_inverse_noise, 
                 pos_special_freqs=[0,-1],
                 freq_noise_c_ell=None,
                 n_components=3, 
                 lmin=2, n_iter=8, 
                 limit_iter_cg=200, 
                 limit_iter_cg_eta=200,
                 tolerance_CG=1e-10, 
                 atol_CG=1e-8,
                 mask=None,
                 save_CMB_chain_maps=False, 
                 save_eta_chain_maps=False,
                 save_all_B_f_params=True,
                 save_s_c_spectra=False,
                 sample_r_Metropolis=True,
                 sample_r_from_BB=False,
                 sample_C_inv_Wishart=False,
                 perturbation_eta_covariance=False,
                 simultaneous_accept_rate=False,
                 non_centered_moves=False,
                 save_intermediary_centered_moves=False,
                 full_sky_correction=False,
                 biased_version=False,
                 classical_Gibbs=False,
                 use_binning=False,
                 bin_ell_distribution=None,
                 acceptance_posdef=False,
                 min_r_to_sample=0,
                 r_true=0,
                 step_size_r=1e-4,
                 covariance_B_f=None,
                 indexes_free_Bf=False,
                 number_iterations_sampling=100, 
                 number_iterations_done=0,
                 seed=0,
                 disable_chex=True,
                 instrument_name='SO_SAT',
                 spv_nodes_b=[]):
        """ 
            Main MICMAC pixel sampling object to initialize and launch the Gibbs sampling in pixel domain. 

            The Gibbs sampling will always store B_f and r (or C) parameters

            Parameters
            ----------
            nside : nside of the input frequency maps, int
            lmax : maximum multipole for the spherical harmonics transforms and harmonic domain objects, int
            nstokes : number of Stokes parameters, int
            frequency_array : array of frequencies, in GHz
            freq_inverse_noise : array of inverse noise for each frequency, in uK^-2
            pos_special_freqs : indexes of the special frequencies in the frequency array respectively for synchrotron and dust, default is [0,-1] for first and last frequencies
            freq_noise_c_ell : optional, noise power spectra for each frequency, in uK^2, dimensions [frequencies, frequencies, lmax+1-lmin] or [frequencies, frequencies, lmax] (in which case it will be cut to lmax+1-lmin)
            n_components : number of components for the mixing matrix, int
            lmin : minimum multipole for the spherical harmonics transforms and harmonic domain objects, int
            n_iter : number of iterations the spherical harmonics transforms (for map2alm transformations), int
            limit_iter_cg : maximum number of iterations for the conjugate gradient for the CMB map sampling, int
            limit_iter_cg_eta : maximum number of iterations for the conjugate gradient for eta maps sampling, int
            tolerance_CG : tolerance for the conjugate gradient, float
            atol_CG : absolute tolerance for the conjugate gradient, float
            mask : optional, mask to use in the sampling, dimensions [nstokes, n_pix] ; if not given, no mask is used
            
            save_CMB_chain_maps : optional, save the CMB chain maps, bool
            save_eta_chain_maps : optional, save the eta chain maps, bool

            sample_r_Metropolis : sample r with Metropolis-Hastings instead of the full power spectrum being sampled, bool
            sample_C_inv_Wishart : sample C_inv with Wishart distribution instead of simply r being sampled, bool

            perturbation_eta_covariance : approach to compute difference between CMB noise component for eta log proba instead of repeating the CG for each B_f sampling, bool
            simultaneous_accept_rate : use the simultaneous accept rate for the patches of the B_f sampling, bool
            
            biased_version : use the biased version of the likelihood, so no computation of the correction term, bool
            classical_Gibbs : sampling only for s_c and the CMB covariance, and neither B_f or eta, bool
            
            use_binning : use binning for the sampling of inverse Wishart CMB covariance, bool
            bin_ell_distribution : binning distribution for the sampling of inverse Wishart CMB covariance, array of integers
            acceptance_posdef : accept only positive definite matrices C sampling, bool
            min_r_to_sample : minimum accepted value for r to sample, float

            r_true : true value of r (only used to compute input CMB maps, not actually used in the sampling), float
            step_size_r : step size for the Metropolis-Hastings sampling of r, float
            covariance_B_f : covariance for the Metropolis-Hastings sampling of B_f, given by a matrix of dimensions [(n_frequencies-len(pos_special_freqs))*(n_components-1), (n_frequencies-len(pos_special_freqs))*(n_components-1)] ; will be repeated if multiresoltion case
            indexes_free_Bf : indexes of the free B_f parameters to actually sample and leave the rest of the indices fixed, array of integers, default False to sample all B_f

            number_iterations_sampling : maximum number of iterations for the sampling, int
            number_iterations_done : number of iterations already accomplished, in case the chain is resuming from a previous run, int

            seed : seed for the JAX PRNG random number generator to start the chain, int
            disable_chex : disable chex tests (to improve speed), bool
            
            instrument_name : optional, name of the instrument, str
            spv_nodes_b : tree for the spatial variability, to generate from a yaml file, list of dictionaries
        """

        ## Give the parameters to the parent class
        super(MICMAC_Sampler,self).__init__(nside=nside,
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
                                            bin_ell_distribution=bin_ell_distribution)
        
        # Run settings
        self.classical_Gibbs = bool(classical_Gibbs) # To run the classical Gibbs sampling instead of the full MICMAC sampling
        if self.classical_Gibbs is False:
            # Then we expect to have multiple components
            assert self.n_components > 1
            try:
                assert len(pos_special_freqs) == self.n_components-1
            except:
                raise Exception("The number of special frequencies should be equal to the number of components - 1")
        self.biased_version = bool(biased_version) # To have a run without the correction term
        self.perturbation_eta_covariance = bool(perturbation_eta_covariance) # To use the perturbation approach for the eta contribution in log-proba of B_f
        self.simultaneous_accept_rate = bool(simultaneous_accept_rate) # To use the simultaneous accept rate for the patches of the B_f sampling
        self.full_sky_correction = bool(full_sky_correction) # To use the full sky correction for the log-proba of Bf sampling
        assert ((sample_r_Metropolis and sample_C_inv_Wishart) == False) and ((sample_r_Metropolis or not(sample_C_inv_Wishart)) or (not(sample_r_Metropolis) or sample_C_inv_Wishart))
        self.sample_r_Metropolis = bool(sample_r_Metropolis)
        assert ((sample_r_from_BB and sample_r_Metropolis)) or not(sample_r_from_BB)
        self.sample_r_from_BB = bool(sample_r_from_BB)
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)
        self.use_binning = bool(use_binning) # To use binning for the sampling of inverse Wishart CMB covariance
        self.acceptance_posdef = bool(acceptance_posdef) # To accept only positive definite matrices for C sampling
        self.non_centered_moves = bool(non_centered_moves) # To use non-centered moves for C sampling
        self.save_intermediary_centered_moves = bool(save_intermediary_centered_moves) # To save intermediary r values in case of non-centered moves in the sampling

        # CMB parameters for input maps generation
        self.r_true = float(r_true)
        
        # Harmonic noise parameter
        self.freq_noise_c_ell = freq_noise_c_ell  # Noise power spectra for each frequency, in uK^2, dimensions [frequencies, frequencies, lmax+1-lmin] or [frequencies, frequencies, lmax] (in which case it will be cut to lmax+1-lmin)

        # Metropolis-Hastings parameters
        self.covariance_B_f = covariance_B_f # Covariance for the Metropolis-Hastings step sampling of B_f
        self.step_size_r = step_size_r # Step size for the Metropolis-Hastings step sampling of r

        # Sampling parameters
        if indexes_free_Bf is False:
            # If given as False, then we sample all B_f
            indexes_free_Bf = jnp.arange(self.len_params)
        self.indexes_free_Bf = jnp.array(indexes_free_Bf)
        assert jnp.size(self.indexes_free_Bf) <= self.len_params # The number of free parameters should be less than the total number of parameters
        assert jnp.isin(self.indexes_free_Bf, jnp.arange(self.len_params)).all() # The indexes should be in the range of the total number of parameters
        self.number_iterations_sampling = int(number_iterations_sampling) # Maximum number of iterations for the sampling
        self.number_iterations_done = int(number_iterations_done) # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed # Seed for the JAX PRNG random number generator to start the chain

        # Saving parameters
        self.save_CMB_chain_maps = bool(save_CMB_chain_maps) # Save the CMB chain maps
        self.save_eta_chain_maps = bool(save_eta_chain_maps) # Save the eta chain maps
        self.save_all_B_f_params = bool(save_all_B_f_params) # Save all the B_f chains
        self.save_s_c_spectra = bool(save_s_c_spectra) # Save the s_c spectra

        # Optional parameters
        self.instrument_name = instrument_name # Name of the instrument
        self.disable_chex = disable_chex # Disable chex tests (to improve speed)

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
        return self.all_samples_wiener_filter_maps +  self.all_samples_fluctuation_maps

    def generate_CMB(self, return_spectra=True):
        """
            Returns CMB spectra of scalar modes only and tensor modes only (with r=1)
            Both CMB spectra are either returned in the usual form [number_correlations,lmax+1], 
            or in the red_cov form if return_spectra == False
        """

        # Selecting the relevant auto- and cross-correlations from CAMB spectra
        if self.nstokes == 2:
            # EE, BB
            partial_indices_polar = np.array([1,2])
        elif self.nstokes == 1:
            # TT
            partial_indices_polar = np.array([0])
        else:
            # TT, EE, BB, EB
            partial_indices_polar = np.arange(4)

        # Generating the CMB power spectra
        all_spectra_r0 = generate_power_spectra_CAMB(self.nside*2, r=0, typeless_bool=True)
        all_spectra_r1 = generate_power_spectra_CAMB(self.nside*2, r=1, typeless_bool=True)

        # Retrieve the scalar mode spectrum
        camb_cls_r0 = all_spectra_r0['total'][:self.lmax+1,partial_indices_polar]

        # Retrieve the tensor mode spectrum
        tensor_spectra_r1 = all_spectra_r1['tensor'][:self.lmax+1,partial_indices_polar]

        theoretical_r1_tensor = np.zeros((self.n_correlations,self.lmax+1))
        theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

        theoretical_r1_tensor[:self.nstokes,...] = tensor_spectra_r1.T
        theoretical_r0_total[:self.nstokes,...] = camb_cls_r0.T

        if return_spectra:
            # Return spectra in the form [number_correlations,lmax+1]
            return theoretical_r0_total, theoretical_r1_tensor

        # Return spectra in the form of the reduced covariance matrix, [lmax+1-lmin,number_correlations,number_correlations]
        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[self.lmin:]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)[self.lmin:]
        return theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor


    def generate_input_freq_maps_from_fgs(self, freq_maps_fgs, return_only_freq_maps=True, return_only_maps=False):
        """
            Generate input frequency maps (CMB+foregrounds) from the input frequency foregrounds maps,
            return either the full frequency maps, the full frequency and CMB maps alone, 
            or the full frequency and CMB maps with the theoretical reduced covariance matrices for the CMB scalar and tensor modes

            Parameters
            ----------
            freq_maps_fgs : input frequency foregrounds maps, dimensions [n_frequencies,nstokes,n_pix]
            return_only_freq_maps : optional, return only the full frequency maps, bool
            return_only_maps : optional, return only the full frequency and CMB maps alone, bool

            Returns
            -------
            input_freq_maps : input frequency maps, dimensions [n_frequencies,nstokes,n_pix]
            input_cmb_maps : input CMB maps, dimensions [nstokes,n_pix]
            theoretical_red_cov_r0_total : theoretical reduced covariance matrix for the CMB scalar modes
            theoretical_red_cov_r1_tensor : theoretical reduced covariance matrix for the CMB tensor modes
        """
        indices_polar = np.array([1,2,4])

        # Generate CMB from CAMB
        theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = self.generate_CMB(return_spectra=False)

        # Retrieve fiducial CMB power spectra
        true_cmb_specra = get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total + self.r_true*theoretical_red_cov_r1_tensor)
        true_cmb_specra_extended = np.zeros((6,self.lmax+1))
        true_cmb_specra_extended[indices_polar,self.lmin:] = true_cmb_specra

        # Generate input frequency maps
        input_cmb_maps_alt = hp.synfast(true_cmb_specra_extended, nside=self.nside, new=True, lmax=self.lmax)[1:,...]
        input_cmb_maps = np.broadcast_to(input_cmb_maps_alt, (self.n_frequencies,self.nstokes,self.n_pix))
        input_freq_maps = input_cmb_maps + freq_maps_fgs

        if return_only_freq_maps:
            return input_freq_maps
        
        if return_only_maps:
            return input_freq_maps, input_cmb_maps

        return input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor

    def update_variable(self, all_samples, new_samples_to_add):
        """
            Update the samples with new samples to add
        """
        if jnp.size(all_samples) == 0:
            return new_samples_to_add
        elif jnp.size(new_samples_to_add.shape) == 1:
            return jnp.hstack([all_samples,new_samples_to_add])
        else:
            return jnp.vstack([all_samples,new_samples_to_add])

    def update_samples(self, all_samples):
        """
            Update the samples with new samples to add
        """
        if self.save_eta_chain_maps and not(self.classical_Gibbs):
            self.all_samples_eta = self.update_variable(self.all_samples_eta, all_samples['eta_maps'])

        if self.save_CMB_chain_maps:
            self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, all_samples['wiener_filter_term'])
            self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, all_samples['fluctuation_maps'])
        
        if self.save_s_c_spectra:
            self.all_samples_s_c_spectra = self.update_variable(self.all_samples_s_c_spectra, all_samples['s_c_spectra'])

        if self.sample_C_inv_Wishart:
            if all_samples['red_cov_matrix_sample'].shape[1] == self.lmax+1-self.lmin:
                all_samples_CMB_c_ell = jnp.array([get_c_ells_from_red_covariance_matrix(all_samples['red_cov_matrix_sample'][iteration]) for iteration in range(self.number_iterations_sampling-self.number_iterations_done)])
            else:
                all_samples_CMB_c_ell = all_samples['red_cov_matrix_sample']
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples_CMB_c_ell)
        if self.sample_r_Metropolis:
            if len(all_samples['r_sample'].shape) != len(self.all_samples_r.shape):
                all_samples['r_sample'] = all_samples['r_sample'].squeeze()
            self.all_samples_r = self.update_variable(self.all_samples_r, all_samples['r_sample'])

        if self.save_all_B_f_params:
            self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, all_samples['params_mixing_matrix_sample'])

    def update_one_sample(self, one_sample):
        """
            Update the samples with one sample to add
        """

        if self.save_eta_chain_maps and not(self.classical_Gibbs):
            self.all_samples_eta = self.update_variable(self.all_samples_eta, jnp.expand_dims(one_sample['eta_maps'],axis=0))

        if self.save_CMB_chain_maps:
            self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, jnp.expand_dims(one_sample['wiener_filter_term'],axis=0))
            self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, jnp.expand_dims(one_sample['fluctuation_maps'],axis=0))

        if self.sample_C_inv_Wishart:
            if one_sample['red_cov_matrix_sample'].shape[0] == self.lmax+1-self.lmin:
                one_sample_CMB_c_ell = get_c_ells_from_red_covariance_matrix(one_sample['red_cov_matrix_sample'])
            else:
                one_sample_CMB_c_ell = one_sample['red_cov_matrix_sample']
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, jnp.expand_dims(one_sample_CMB_c_ell,axis=0))
        if self.sample_r_Metropolis:
            if self.non_centered_moves:
                if self.save_intermediary_centered_moves:
                    self.all_samples_r = self.update_variable(self.all_samples_r, jnp.expand_dims(jnp.stack((one_sample['r_sample'],one_sample['r_sample'])),axis=0))
                else:
                    self.all_samples_r = self.update_variable(self.all_samples_r, one_sample['r_sample'],axis=0)
            else:
                self.all_samples_r = self.update_variable(self.all_samples_r, one_sample['r_sample'])

        if self.save_all_B_f_params:
            self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, jnp.expand_dims(one_sample['params_mixing_matrix_sample'],axis=0))


    def perform_Gibbs_sampling(self, 
                               input_freq_maps, 
                               c_ell_approx, 
                               CMB_c_ell, 
                               init_params_mixing_matrix, 
                               initial_guess_r=0, 
                               initial_wiener_filter_term=jnp.empty(0), 
                               initial_fluctuation_maps=jnp.empty(0),
                               theoretical_r0_total=jnp.empty(0), 
                               theoretical_r1_tensor=jnp.empty(0),
                               suppress_low_modes_input_freq_maps=False):
        """ 
            Perform sampling steps with :
                1. The sampling of \eta by computing \eta =  x + C_approx^(1/2) N_c^{-1/2} y
                2. A CG for the Wiener filter (WF) and fluctuation variables s_c : (s_c - s_{c,WF})^t (C^{-1} + N_c^{-1}) (s_c - s_{c,WF})
                3. The c_ell sampling, either by parametrizing it by r or by sampling an inverse Wishart distribution 
                4. Mixing matrix B_f sampling with : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + \eta^t (Id + C_{approx}^{1/2} N_c^{-1} C_{approx}^{1/2}) \eta

            The results of the chain will be stored in the class attributes, depending if the save options are put to True or False:
                - self.all_samples_eta (if self.save_eta_chain_maps is True)
                - self.all_samples_wiener_filter_maps (if self.save_CMB_chain_maps is True)
                - self.all_samples_fluctuation_maps (if self.save_CMB_chain_maps is True)
                - self.all_samples_r (if self.sample_r_Metropolis is True)
                - self.all_samples_CMB_c_ell (if self.sample_C_inv_Wishart is True)
                - self.all_params_mixing_matrix_samples (always)
            
            This same function can be used to continue a chain from a previous run, by giving the number of iterations already done in the MICMAC_Sampler object,
            giving the chains to the attributes of the object, and giving the last iteration results as initial guesses.

            Parameters
            ----------
            input_freq_maps : input frequency maps, dimensions [frequencies, nstokes, n_pix]
            c_ell_approx : approximate CMB power spectra for the correction term, dimensions [number_correlations, lmax+1]
            CMB_c_ell : CMB power spectra, dimensions [number_correlations, lmax+1]
            init_params_mixing_matrix : initial parameters for the mixing matrix, dimensions [len_params]
            initial_guess_r : optional, initial guess for r, float
            initial_wiener_filter_term : optional, initial guess for the Wiener filter term, dimensions [nstokes, n_pix]
            initial_fluctuation_maps : optional, initial guess for the fluctuation maps, dimensions [nstokes, n_pix]
            theoretical_r0_total : optional, theoretical reduced covariance matrix for the CMB scalar modes, dimensions [number_correlations, lmax+1-lmin]
            theoretical_r1_tensor : optional, theoretical reduced covariance matrix for the CMB tensor modes, dimensions [number_correlations, lmax+1-lmin]
            suppress_low_modes_input_freq_maps : optional, suppress low modes in the input frequency maps, bool


        """

        # Disabling all chex checks to speed up the code
        if self.disable_chex:
            print("Disabling chex !!!", flush=True)
            chx.disable_asserts()

        ## Getting only the relevant spectra
        if self.nstokes == 2:
            indices_to_consider = np.array([1,2,4])
            partial_indices_polar = indices_to_consider[:self.nstokes]
        elif self.nstokes == 1:
            indices_to_consider = np.array([0])
        else:
            indices_to_consider = np.arange(6) # All auto- and cross-correlations

        ## Testing the initial WF term, or initialize it properly
        if len(initial_wiener_filter_term) == 0:
            wiener_filter_term = jnp.zeros((self.nstokes,self.n_pix))
        else:
            assert len(initial_wiener_filter_term.shape) == 2
            assert initial_wiener_filter_term.shape == (self.nstokes, self.n_pix)
            # assert initial_wiener_filter_term.shape[1] == self.n_pix
            wiener_filter_term = initial_wiener_filter_term

        ## Testing the initial fluctuation term, or initialize it properly
        if len(initial_fluctuation_maps) == 0:
            fluctuation_maps = jnp.zeros((self.nstokes,self.n_pix))
        else:
            assert len(initial_fluctuation_maps.shape) == 2
            assert initial_fluctuation_maps.shape == (self.nstokes, self.n_pix)
            # assert initial_fluctuation_maps.shape[1] == self.n_pix
            fluctuation_maps = initial_fluctuation_maps

        ## Testing the initial spectra given in case the sampling is done with r
        if self.sample_r_Metropolis:
            assert len(theoretical_r0_total.shape) == 2
            assert (theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin)
            assert len(theoretical_r1_tensor.shape) == 2
            assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

            # Transforming into the reduced (red) format [lmax+1-lmin,nstokes,nstokes]
            theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
            theoretical_red_cov_r1_tensor  = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)
            assert theoretical_red_cov_r0_total.shape[1] == self.nstokes

        ## Testing the initial CMB spectra and C_approx spectra given
        if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):
            CMB_c_ell = CMB_c_ell[indices_to_consider,:]
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[indices_to_consider,:]
        
        assert len(CMB_c_ell.shape) == 2
        assert CMB_c_ell.shape[1] == self.lmax + 1
        assert len(c_ell_approx.shape) == 2
        assert c_ell_approx.shape[1] == self.lmax + 1
        
        ## Testing the initial mixing matrix
        if self.n_components != 1:
            assert init_params_mixing_matrix.shape == (self.len_params,)

        ## Testing the input frequency maps
        assert len(input_freq_maps.shape) == 3
        assert input_freq_maps.shape == (self.n_frequencies, self.nstokes, self.n_pix)

        ## Testing the mask
        assert np.abs(self.mask).sum() != 0


        ## Testing the initial guess for r
        assert np.size(initial_guess_r) == 1
        # assert initial_guess_r >= 0 # Not allowing for first guess negative r values

        # Preparing for the full Gibbs sampling
        len_pos_special_freqs = len(self.pos_special_freqs)

        # input_freq_maps_ = jnp.array(input_freq_maps)
        # if suppress_low_modes_input_freq_maps:
        #     print("Suppressing out of band [lmin,lmax] modes in the input frequency maps !", flush=True)
        #     def fmap(index):
        #         return self.get_band_limited_maps(input_freq_maps_[index])
        #     input_freq_maps = jax.vmap(fmap)(jnp.arange(self.n_frequencies))
        # del input_freq_maps_

        if self.use_binning:
            print("Using binning for the sampling of CMB covariance !!!", flush=True)
            print("Binning distribution :", self.bin_ell_distribution, flush=True)

        ## Initial guesses preparation
        
        ## eta
        initial_eta = jnp.zeros((self.nstokes,self.n_pix))
        ## CMB covariance preparation in the format [lmax,nstokes,nstokes]
        red_cov_approx_matrix = jnp.array(get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin:,...])
        red_cov_matrix = get_reduced_matrix_from_c_ell(CMB_c_ell)[self.lmin:,...]
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
        # r_sampling_MH = single_Metropolis_Hasting_step
        r_sampling_MH = bounded_single_Metropolis_Hasting_step
        if self.sample_r_Metropolis:
            log_proba_r = self.get_conditional_proba_C_from_r
            if self.sample_r_from_BB:
                print("Sampling r from BB !!!", flush=True)
                log_proba_r = self.get_conditional_proba_C_from_r_wBB
            if self.use_binning:
                print("Using BB binning for the sampling of r !!!", flush=True)
                log_proba_r = self.get_binned_conditional_proba_C_from_r_wBB

        ## Function to sample the mixing matrix free parameters in the most general way
        jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v2b_JAX, static_argnames=['biased_bool', 'full_sky_correction'])
        sampling_func = separate_single_MH_step_index_accelerated

        if self.biased_version or self.perturbation_eta_covariance:
            print("Using biased version or perturbation version of mixing matrix sampling !!!", flush=True)
            ## Function to sample the mixing matrix free parameters through the difference of the log-proba, to have only one CG done
            if self.full_sky_correction:
                print("Using full_sky correction !!!", flush=True)
            jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v3_JAX, static_argnames=['biased_bool', 'full_sky_correction'])
            sampling_func = separate_single_MH_step_index_v2b

            if self.simultaneous_accept_rate:
                ## More efficient version of the mixing matrix sampling
                
                ## MH step function to sample the mixing matrix free parameters with patches simultaneous computed accept rate
                print("Using simultaneous accept rate version of mixing matrix sampling !!!", flush=True)
                print("---- ATTENTION : This assumes all patches are distributed in the same way for all parameters !", flush=True)
                jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v3_pixel_JAX, static_argnames=['biased_bool', 'full_sky_correction'])
                sampling_func = separate_single_MH_step_index_v4_pixel
                if (self.size_patches != self.size_patches[0]).any():
                    sampling_func = separate_single_MH_step_index_v4b_pixel
                    # raise NotImplemented("All patches should have the same size for the simultaneous accept rate version of mixing matrix sampling for now !!!")

                ## Redefining the free Bf indexes to sample to the one 
                # condition_unobserved_patches = self.get_cond_unobserved_patches() ## Get boolean array to identify which free indexes are not relevant
                # print("Previous free indexes for B_f", self.indexes_free_Bf, flush=True)
                # self.indexes_free_Bf = jnp.array(self.indexes_free_Bf).at[condition_unobserved_patches].get()
                # print("New free indexes for B_f", self.indexes_free_Bf, flush=True)

                print("Previous free indexes for B_f", self.indexes_free_Bf, self.indexes_free_Bf.size, flush=True)
                self.indexes_free_Bf = self.indexes_free_Bf.at[self.get_cond_unobserved_patches_from_indices(self.indexes_free_Bf)].get() 
                ## Get boolean array to identify which free indexes are not relevant
                print("New free indexes for B_f", self.indexes_free_Bf, self.indexes_free_Bf.size, flush=True)

                indexes_patches_Bf = jnp.array(self.indexes_b.ravel(order='F'), dtype=jnp.int64)
                def which_interval(carry, index_Bf):
                    """
                        Selecting the patches to be used for the B_f sampling by checking if the index_Bf is in the interval of the patches
                    """
                    return carry | ((index_Bf >= indexes_patches_Bf) & (index_Bf < indexes_patches_Bf + self.size_patches)), index_Bf

                condition, _ = jlax.scan(which_interval, jnp.zeros_like(self.size_patches, dtype=bool), self.indexes_free_Bf)
                
                first_indices_patches_free_Bf = indexes_patches_Bf[condition]
                max_len_patches_Bf = int(np.max(self.size_patches[condition]))
                size_patches = self.size_patches[condition]


        ## Preparing the random JAX PRNG key
        if np.size(self.seed) == 1:
            PRNGKey = random.PRNGKey(self.seed)
        elif np.size(self.seed) == 2:
            PRNGKey = jnp.array(self.seed, dtype=jnp.uint32)
        else:
            raise ValueError("Seed should be either a scalar or a 2D array interpreted as a JAX PRNG Key!")

        ## Computing the number of iterations to perform
        actual_number_of_iterations = self.number_iterations_sampling - self.number_iterations_done

        

        
        if not(self.classical_Gibbs):
            ## Preparing the step-size for Metropolis-within-Gibbs of B_f sampling

            ## try/except step only because jsp.linalg.sqrtm is not implemented in GPU
            try :
                initial_step_size_Bf = jnp.array(jnp.diag(jsp.linalg.sqrtm(self.covariance_B_f)), dtype=jnp.float64)
            except:
                initial_step_size_Bf = jnp.array(jnp.diag(jnp.sqrt(self.covariance_B_f)), dtype=jnp.float64)
            assert len(initial_step_size_Bf.shape) == 1
            print('Step-size B_f', initial_step_size_Bf, flush=True)
            if self.covariance_B_f.shape[0] != self.len_params:
                print("Covariance matrix for B_f is not of the right shape !", flush=True)
                # initial_step_size_Bf = jnp.repeat(initial_step_size_Bf, self.len_params//self.covariance_B_f.shape[0], axis=0)
                initial_step_size_Bf = jnp.broadcast_to(initial_step_size_Bf, 
                                                        (self.len_params//self.covariance_B_f.shape[0],self.covariance_B_f.shape[0])
                                                        ).ravel(order='F')
                print('New step-size B_f', initial_step_size_Bf, flush=True)

        ## Few prints to re-check the toml parameters chosen
        if self.classical_Gibbs:
            print("Not sampling for eta and B_f, only for s_c and the CMB covariance !", flush=True)
        if self.sample_r_Metropolis:
            print("Sample for r instead of C !", flush=True)
            if self.sample_r_from_BB:
                print("Sample for r with the BB likelihood !", flush=True)
        if self.non_centered_moves:
            print("Using non-centered moves for C sampling !", flush=True)
            if self.save_intermediary_centered_moves:
                print("Saving intermediary centered moves for C sampling !", flush=True)

        else:
            print("Sample for C with inverse Wishart !", flush=True)

        # Few steps to improve the speed of the code

        ## Preparing the square root matrix of C_approx
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)

        ## Preparing the preconditioner in the case of a full sky and white noise
        use_precond = False
        if self.mask.sum() == self.n_pix and self.freq_noise_c_ell is not None:
            assert len(self.freq_noise_c_ell.shape) == 3
            assert self.freq_noise_c_ell.shape[0] == self.n_frequencies
            assert self.freq_noise_c_ell.shape[1] == self.n_frequencies
            assert (self.freq_noise_c_ell.shape[2] == self.lmax+1) or (self.freq_noise_c_ell.shape[2] == self.lmax+1-self.lmin)
            if self.freq_noise_c_ell.shape[2] == self.lmax+1:
                self.freq_noise_c_ell = self.freq_noise_c_ell[...,self.lmin:]
            self.freq_noise_c_ell = jnp.array(self.freq_noise_c_ell)

            print("Full sky case !", flush=True)
            use_precond = True

        ## Finally starting the Gibbs sampling !!!
        print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done", flush=True)

        @scan_tqdm(actual_number_of_iterations, )
        def all_sampling_steps(carry, iteration):
            """ 
                1-step Gibbs sampling function, performing the following:
                - Sampling of eta, for the correction term ; perform as well a CG if the perturbation approach is chosen
                - Sampling of s_c, for the constrained CMB map realization ; sampling both Wiener filter and fluctuation maps
                - Sampling of C or r parametrizing C, for the CMB covariance matrix
                - Sampling of the free B_f parameters, for the mixing matrix

                Parameters
                ----------
                :param carry: dictionnary containing the following variables at 1 iteration depending on the option chosen : WF maps, fluctuation maps, CMB covariance, r samples, B_f samples, PRNGKey
                :param iteration: current iteration number

                Returns
                -------
                :return new_carry: dictionnary containing the following variables at the next iteration : WF maps, fluctuation maps, CMB covariance, r sample, B_f sample, PRNGKey
                :return all_samples: dictionnary containing the variables to save as chains, so depending on the options chosen: eta maps, WF maps, fluctuation maps, CMB covariance, r sample, B_f sample
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

            # Sampling step 1 : sampling of Gaussian variable eta
            
            ## Initialize the preconditioner for the eta contribution
            precond_func_eta = None

            ## Sampling of eta if not using the classical Gibbs sampling and neither the biased version
            if not(self.classical_Gibbs) and not(self.biased_version):
                # Preparing random variables
                map_random_x = None
                map_random_y = None

                # Sampling eta maps
                new_carry['eta_maps'] = self.get_sampling_eta_v2(red_cov_approx_matrix_sqrt, invBtinvNB, BtinvN_sqrt, 
                                                               subPRNGKey, map_random_x=map_random_x, map_random_y=map_random_y, 
                                                               suppress_low_modes=True)

                # Checking shape of the resulting maps
                chx.assert_shape(new_carry['eta_maps'], (self.nstokes, self.n_pix))

                # Preparing the preconditioner for the CG
                if use_precond:
                    ## Assuming a harmonic noise with the pixel average of the mixing matrix
                    noise_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled.mean(axis=2))[0,0]
                    ## Getting N_c^{-1} for the harmonic noise covariance
                    red_inv_noise_c_ell = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(jnp.stack([noise_c_ell, noise_c_ell, jnp.zeros_like(noise_c_ell)])))
                    red_preconditioner_eta = jnp.linalg.pinv(jnp.eye(self.nstokes) 
                                                            + jnp.einsum('lij,ljk,lkm->lim', 
                                                            red_cov_approx_matrix_sqrt, 
                                                            red_inv_noise_c_ell, 
                                                            red_cov_approx_matrix_sqrt))
                    precond_func_eta = lambda x: maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), 
                                                                                red_preconditioner_eta, 
                                                                                nside=self.nside, 
                                                                                lmin=self.lmin, 
                                                                                n_iter=self.n_iter).ravel()

                if self.perturbation_eta_covariance:
                    # Computing the inverse associated log proba term fixed correction covariance for the B_f sampling, in case of the perturbative approach
                    _, inverse_term = func_logproba_eta(invBtinvNB[0,0],
                                                        new_carry['eta_maps'], 
                                                        red_cov_approx_matrix_sqrt, 
                                                        first_guess=carry['inverse_term'], 
                                                        return_inverse=True,
                                                        precond_func=precond_func_eta)
                else:
                    inverse_term = carry['inverse_term']

                if self.save_eta_chain_maps:
                    all_samples['eta_maps'] = new_carry['eta_maps']

            # Sampling step 2 : sampling of Gaussian variable s_c, contrained CMB map realization

            ## Geting the square root matrix of the sampled CMB covariance
            red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(carry['red_cov_matrix_sample'])

            # Preparing the preconditioner to use for the sampling of the CMB maps
            precond_func_s_c = None
            if use_precond:
                ## Assuming a harmonic noise with the pixel average of the mixing matrix
                noise_c_ell = get_inv_BtinvNB_c_ell(self.freq_noise_c_ell, mixing_matrix_sampled.mean(axis=2))[0,0]
                ## Getting N_c^{-1} for the harmonic noise covariance
                red_inv_noise_c_ell = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(jnp.stack([noise_c_ell, noise_c_ell, jnp.zeros_like(noise_c_ell)])))#[self.lmin:]
                red_preconditioner_s_c = jnp.linalg.pinv(jnp.eye(self.nstokes) 
                                                        + jnp.einsum('lij,ljk,lkm->lim', 
                                                        red_cov_matrix_sqrt, 
                                                        red_inv_noise_c_ell, 
                                                        red_cov_matrix_sqrt))

                precond_func_s_c = lambda x: maps_x_red_covariance_cell_JAX(x.reshape((self.nstokes,self.n_pix)), 
                                                                            red_preconditioner_s_c, 
                                                                            nside=self.nside, 
                                                                            lmin=self.lmin, 
                                                                            n_iter=self.n_iter).ravel()
            
            ## Computing an initial guess closer to the actual start of the CG for the Wiener filter
            initial_guess_WF = maps_x_red_covariance_cell_JAX(carry['wiener_filter_term'], 
                                                              jnp.linalg.pinv(red_cov_matrix_sqrt), 
                                                              nside=self.nside, 
                                                              lmin=self.lmin, 
                                                              n_iter=self.n_iter)
            ## Sampling the Wiener filter term
            new_carry['wiener_filter_term'] = sampling_func_WF(s_cML, 
                                                               red_cov_matrix_sqrt, 
                                                               invBtinvNB, 
                                                               initial_guess=initial_guess_WF,
                                                               precond_func=precond_func_s_c)

            ## Preparing the random variables for the fluctuation term
            PRNGKey, new_subPRNGKey = random.split(PRNGKey)
            map_random_realization_xi = None
            map_random_realization_chi = None

            ## Getting the fluctuation maps terms, for the variance of the variable s_c
            initial_guess_Fluct = maps_x_red_covariance_cell_JAX(carry['fluctuation_maps'], 
                                                                jnp.linalg.pinv(red_cov_matrix_sqrt), 
                                                                nside=self.nside, 
                                                                lmin=self.lmin, 
                                                                n_iter=self.n_iter)
            ## Sampling the fluctuation maps
            new_carry['fluctuation_maps'] = sampling_func_Fluct(red_cov_matrix_sqrt, 
                                                                invBtinvNB, 
                                                                BtinvN_sqrt, 
                                                                new_subPRNGKey, 
                                                                map_random_realization_xi=map_random_realization_xi, 
                                                                map_random_realization_chi=map_random_realization_chi, 
                                                                initial_guess=initial_guess_Fluct,
                                                                precond_func=precond_func_s_c)

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


            # Sampling step 3 : sampling of CMB covariance C
            
            ## Preparing the c_ell which will be used for the sampling
            c_ells_Wishart_ = get_cell_from_map_jax(s_c_sample, lmax=self.lmax, n_iter=self.n_iter)[:,self.lmin:]

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
                new_carry['red_cov_matrix_sample'] = func_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_, 
                                                                    PRNGKey=new_subPRNGKey_2, 
                                                                    old_sample=carry['red_cov_matrix_sample'], 
                                                                    acceptance_posdef=self.acceptance_posdef)
                all_samples['red_cov_matrix_sample'] = new_carry['red_cov_matrix_sample']

            elif self.sample_r_Metropolis:
                # Sampling r which will parametrize C(r) = C_scalar + r*C_tensor
                new_carry['r_sample'] = r_sampling_MH(random_PRNGKey=new_subPRNGKey_2, old_sample=carry['r_sample'],
                                                      step_size=self.step_size_r, log_proba=log_proba_r, 
                                                      red_sigma_ell=red_c_ells_Wishart_modified, 
                                                      theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, 
                                                      theoretical_red_cov_r0_total=theoretical_red_cov_r0_total,
                                                      min_value=self.min_r_to_sample)

                ## Reconstructing the new spectra from r
                new_carry['red_cov_matrix_sample'] = theoretical_red_cov_r0_total + new_carry['r_sample']*theoretical_red_cov_r1_tensor

                ## Binning if needed
                if self.use_binning:
                    new_carry['red_cov_matrix_sample'] = self.bin_and_reproject_red_c_ell(new_carry['red_cov_matrix_sample'])

                ## Saving the r sample
                all_samples['r_sample'] = new_carry['r_sample']
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')
            
            
            if self.non_centered_moves:
                PRNGKey, new_subPRNGKey_2b = random.split(PRNGKey)
                if self.sample_r_Metropolis:
                    new_r_sample = r_sampling_MH(random_PRNGKey=new_subPRNGKey_2b, old_sample=new_carry['r_sample'],
                                                        step_size=self.step_size_r, log_proba=self.get_log_proba_non_centered_move_C_from_r,
                                                        old_r_sample=new_carry['r_sample'],
                                                        invBtinvNB=invBtinvNB,
                                                        s_cML=s_cML,
                                                        s_c_sample=s_c_sample, 
                                                        theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, 
                                                        theoretical_red_cov_r0_total=theoretical_red_cov_r0_total,
                                                        min_value=self.min_r_to_sample)

                    new_carry['red_cov_matrix_sample'] = theoretical_red_cov_r0_total + new_carry['r_sample']*theoretical_red_cov_r1_tensor

                    if self.save_intermediary_centered_moves:
                        all_samples['r_sample'] = jnp.stack((new_carry['r_sample'], new_r_sample))
                    else:
                        all_samples['r_sample'] = new_r_sample

                    new_carry['r_sample'] = new_r_sample

            ## Checking the shape of the resulting covariance matrix, and correcting it if needed
            if new_carry['red_cov_matrix_sample'].shape[0] == self.lmax + 1:
                new_carry['red_cov_matrix_sample'] = new_carry['red_cov_matrix_sample'][self.lmin:]

            ## Small check on the shape of the resulting covariance matrix
            chx.assert_shape(new_carry['red_cov_matrix_sample'], (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))


            # Sampling step 4 : sampling of mixing matrix B_f

            ## Preparation of sampling step 4

            ## First preparing the term: d - B_c s_c
            full_data_without_CMB = input_freq_maps - jnp.broadcast_to(s_c_sample, (self.n_frequencies, self.nstokes, self.n_pix))
            chx.assert_shape(full_data_without_CMB, (self.n_frequencies, self.nstokes, self.n_pix))

            ## Preparing the new PRNGKey
            PRNGKey, new_subPRNGKey_3 = random.split(PRNGKey)
            
            ## Performing the sampling
            if not(self.classical_Gibbs):
                # Preparing the step-size
                step_size_Bf = initial_step_size_Bf

                # Sampling B_f
                if self.perturbation_eta_covariance or self.biased_version:
                    

                    ## Preparing the parameters to provide for the sampling of B_f
                    dict_parameters_sampling_B_f = {'indexes_Bf':self.indexes_free_Bf,
                                                    'full_data_without_CMB':full_data_without_CMB, 
                                                    'red_cov_approx_matrix_sqrt':red_cov_approx_matrix_sqrt, 
                                                    'old_params_mixing_matrix':carry['params_mixing_matrix_sample'],
                                                    'biased_bool':self.biased_version}
                    if self.perturbation_eta_covariance:
                        ## Precomputing the term C_approx^{1/2} A^{-1} eta = C_approx^{1/2} ( Id + C_approx^{1/2} N_{c,old}^{-1} C_approx^{1/2} )^{-1} eta
                        inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.n_pix), 
                                                                                    red_cov_approx_matrix_sqrt, 
                                                                                    nside=self.nside, 
                                                                                    lmin=self.lmin, 
                                                                                    n_iter=self.n_iter
                                                                                    ).ravel()
                        dict_parameters_sampling_B_f['previous_inverse_x_Capprox_root'] = inverse_term_x_Capprox_root
                        dict_parameters_sampling_B_f['first_guess'] = inverse_term

                    if not(self.biased_version):
                        ## If not biased, provide the eta maps
                        dict_parameters_sampling_B_f['component_eta_maps'] = new_carry['eta_maps']
                    if self.simultaneous_accept_rate:
                        ## Provide as well the indexes of the patches in case of the uncorrelated patches version
                        dict_parameters_sampling_B_f['size_patches'] = size_patches
                        dict_parameters_sampling_B_f['max_len_patches_Bf'] = max_len_patches_Bf
                        dict_parameters_sampling_B_f['indexes_patches_Bf'] = first_indices_patches_free_Bf
                        dict_parameters_sampling_B_f['len_indexes_Bf'] = self.len_params
                        #TODO: Accelerate by removing indexes of indexes_patches_Bf if the corresponding patches are not in indexes_free_Bf, nor in the mask
                    ## Test parameter
                    dict_parameters_sampling_B_f['full_sky_correction'] = self.full_sky_correction
                    ## Sampling B_f !
                    new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'] = sampling_func(random_PRNGKey=new_subPRNGKey_3, 
                                                                                                old_sample=carry['params_mixing_matrix_sample'], 
                                                                                                step_size=step_size_Bf,
                                                                                                log_proba=jitted_Bf_func_sampling,
                                                                                                **dict_parameters_sampling_B_f)
                else:
                    ## Sampling B_f with older version -> might be slower
                    new_subPRNGKey_3, new_carry['params_mixing_matrix_sample'], inverse_term = sampling_func(random_PRNGKey=new_subPRNGKey_3, old_sample=carry['params_mixing_matrix_sample'], 
                                                            step_size=step_size_Bf, indexes_Bf=self.indexes_free_Bf,
                                                            log_proba=jitted_Bf_func_sampling,
                                                            full_data_without_CMB=full_data_without_CMB, component_eta_maps=new_carry['eta_maps'], 
                                                            red_cov_approx_matrix_sqrt=red_cov_approx_matrix_sqrt,
                                                            first_guess=carry['inverse_term'],
                                                            biased_bool=self.biased_version,
                                                            precond_func=precond_func_eta)
                if self.perturbation_eta_covariance:
                    ## Passing the inverse term to the next iteration
                    new_carry['inverse_term'] = inverse_term

                # Checking the shape of the resulting mixing matrix
                chx.assert_shape(new_carry['params_mixing_matrix_sample'], (self.len_params,))
            else:
                ## Classical Gibbs sampling, no need to sample B_f but still needs to provide them to the next iteration in case it is used for the CMB noise component
                new_carry['params_mixing_matrix_sample'] = carry['params_mixing_matrix_sample']
                # all_samples['params_mixing_matrix_sample'] = new_carry['params_mixing_matrix_sample']
            
            ## Saving the B_f obtained
            all_samples['params_mixing_matrix_sample'] = new_carry['params_mixing_matrix_sample']

            ## Passing as well the PRNGKey to the next iteration
            new_carry['PRNGKey'] = PRNGKey
            return new_carry, all_samples

        ## Preparing the initial carry
        initial_carry = {'wiener_filter_term': wiener_filter_term,
                            'fluctuation_maps': fluctuation_maps,
                            'red_cov_matrix_sample': red_cov_matrix,
                            'params_mixing_matrix_sample': params_mixing_matrix_init_sample,
                            'PRNGKey': PRNGKey}

        if not(self.classical_Gibbs) and not(self.biased_version):
            initial_carry['eta_maps'] = initial_eta
        if not(self.classical_Gibbs) and not(self.biased_version):
            initial_carry['inverse_term'] = jnp.zeros_like(initial_eta)
        if self.sample_r_Metropolis:
            initial_carry['r_sample'] = initial_guess_r
        if self.save_s_c_spectra:
            self.all_samples_s_c_spectra = self.update_variable(self.all_samples_s_c_spectra, jnp.expand_dims(jnp.zeros((self.n_correlations, self.lmax+1-self.lmin)),axis=0))

        ## Initialising the first carry to the chains saved
        self.update_one_sample(initial_carry)

        ## Starting the Gibbs sampling !!!!
        time_start_sampling = time.time()
        # Start sampling !!!
        last_sample, all_samples = jlx.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        time_full_chain = (time.time()-time_start_sampling)/60      
        print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

        
        # Saving the samples as attributes of the Sampler object
        time_start_updating = time.time()
        self.update_samples(all_samples)
        time_end_updating = (time.time()-time_start_updating)/60
        print("End of updating in {} minutes".format(time_end_updating), flush=True)
        self.number_iterations_done = self.number_iterations_sampling

        print("Last key PRNG", last_sample['PRNGKey'], flush=True)
        self.last_PRNGKey = last_sample['PRNGKey']

        ## Saving the last sample
        self.last_sample = last_sample


def create_MICMAC_sampler_from_toml_file(path_toml_file, path_file_spv=''):
    """
        Create a MICMAC_Sampler object from:
        * the path of a toml file: params for the sims and for the sampling
        * the path of a spv file: params for addressing spatial variability
    """
    ### Opening first the toml file for the simulations and sampling, to create the MICMAC_Sampler object
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    ## Getting the instrument and the noise covariance
    if dictionary_parameters['instrument_name'] != 'customized_instrument': ## TODO: Improve a bit this part
        instrument = fgbuster.get_instrument(dictionary_parameters['instrument_name'])
        
    else:
        instrument = get_instr(dictionary_parameters['frequency_array'], dictionary_parameters['depth_p'])
        del dictionary_parameters['depth_p']
        

    dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
    dictionary_parameters['freq_inverse_noise'] = get_noise_covar_extended(instrument['depth_p'], dictionary_parameters['nside'])

    ## Spatial variability (spv) params
    n_fgs_comp = dictionary_parameters['n_components']-1
    # total number of params in the mixing matrix for a specific pixel
    n_betas = (np.shape(dictionary_parameters['frequency_array'])[0]-len(dictionary_parameters['pos_special_freqs']))*(n_fgs_comp)
    # Read or create spv config
    root_tree = tree_spv_config(path_file_spv, n_betas, n_fgs_comp, print_tree=True)
    dictionary_parameters['spv_nodes_b'] = get_nodes_b(root_tree)

    return MICMAC_Sampler(**dictionary_parameters)
