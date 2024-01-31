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
from .algorithm_toolbox import *
from .proba_functions import *
# from .Sampling_toolbox import *
from .mixingmatrix import *
from .noisecovar import *
from .jax_Sampling_toolbox import *
from .temporary_tools import *

from jax import config
config.update("jax_enable_x64", True)

class MICMAC_Sampler(Sampling_functions):
    def __init__(self, nside, lmax, nstokes, 
                 frequency_array, freq_inverse_noise, pos_special_freqs=[0,-1], 
                 number_components=3, lmin=2,
                 lmin_r=-1, lmax_r=-1,
                 n_iter=8, limit_iter_cg=2000, tolerance_CG=1e-10, atol_CG=1e-8,
                 limit_iter_cg_eta=200,
                 mask=None,
                 use_automatic_step_size=False, num_sample_AM = 100000000, 
                 epsilon_cov = 1e-20, scale_param = 2.38**2,

                 restrict_to_mask=False,
                 bin_ell_distribution=None,
                 use_old_s_c_sampling=False,
                 perturbation_eta_covariance=False,
                 acceptance_posdef=False,

                 use_binning=False,
                 cheap_save=True, very_cheap_save=False,
                 biased_version=False, lognormal_r=False,
                 r_true=0,
                 sample_eta_B_f=True,
                 sample_r_Metropolis=True, sample_C_inv_Wishart=False,
                 step_size_B_f=1e-5, step_size_r=1e-4,
                 indexes_free_Bf=False,

                 instrument_name='SO_SAT',
                 number_iterations_sampling=100, number_iterations_done=0, seed=0,
                 disable_chex=True):
        """ Non parametric likelihood sampling object
        """

        super(MICMAC_Sampler,self).__init__(nside=nside,lmax=lmax,nstokes=nstokes,lmin=lmin,
                                            lmin_r=lmin_r, lmax_r=lmax_r,
                                            frequency_array=frequency_array,freq_inverse_noise=freq_inverse_noise, 
                                            pos_special_freqs=pos_special_freqs,number_components=number_components,
                                            n_iter=n_iter, limit_iter_cg=limit_iter_cg, limit_iter_cg_eta=limit_iter_cg_eta, 
                                            tolerance_CG=tolerance_CG, atol_CG=atol_CG, 
                                            mask=mask, restrict_to_mask=restrict_to_mask, bin_ell_distribution=bin_ell_distribution)

        # Quick test parameters
        self.instrument_name = instrument_name
        self.sample_eta_B_f = bool(sample_eta_B_f)
        self.biased_version = bool(biased_version)
        self.cheap_save = bool(cheap_save)
        self.very_cheap_save = bool(very_cheap_save)
        self.disable_chex = disable_chex
        if indexes_free_Bf is False:
            indexes_free_Bf = jnp.arange((self.number_frequencies-len(pos_special_freqs))*(self.number_correlations-1))
        self.indexes_free_Bf = jnp.array(indexes_free_Bf)
        self.use_old_s_c_sampling = bool(use_old_s_c_sampling)
        # self.fixed_eta_covariance = bool(fixed_eta_covariance)
        self.perturbation_eta_covariance = bool(perturbation_eta_covariance)
        self.use_binning = bool(use_binning)
        self.acceptance_posdef = bool(acceptance_posdef)

        # CMB parameters
        self.r_true = float(r_true)
        assert ((sample_r_Metropolis and sample_C_inv_Wishart) == False) and ((sample_r_Metropolis or not(sample_C_inv_Wishart)) or (not(sample_r_Metropolis) or sample_C_inv_Wishart))
        self.sample_r_Metropolis = bool(sample_r_Metropolis)
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)


        # Metropolis-Hastings parameters
        self.lognormal_r = lognormal_r
        # self.step_size_B_f = step_size_B_f
        self.covariance_step_size_B_f = jnp.diag((jnp.ravel(step_size_B_f,order='F')**2)*jnp.ones((self.number_frequencies-len(pos_special_freqs))*(self.number_correlations-1)))
        self.step_size_r = step_size_r
        # self.number_steps_sampler_B_f = int(number_steps_sampler_B_f) # Maximum number of steps for the Metropolis-Hasting to sample the mixing matrix
        # self.number_steps_sampler_r = int(number_steps_sampler_r)
        self.use_automatic_step_size = use_automatic_step_size
        self.num_sample_AM = num_sample_AM
        self.epsilon_cov = epsilon_cov
        self.scale_param = scale_param

        # Sampling parameters
        self.number_iterations_sampling = int(number_iterations_sampling) # Maximum number of iterations for the sampling
        self.number_iterations_done = int(number_iterations_done) # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed

        # Samples preparation
        self.all_samples_eta = jnp.empty(0)
        self.all_params_mixing_matrix_samples = jnp.empty(0)
        self.all_samples_wiener_filter_maps = jnp.empty(0)
        self.all_samples_fluctuation_maps = jnp.empty(0)
        self.all_samples_r = jnp.empty(0)
        self.all_samples_CMB_c_ell = jnp.empty(0)
    

    @property
    def all_samples_s_c(self):
        return self.all_samples_wiener_filter_maps +  self.all_samples_fluctuation_maps



    def generate_CMB(self, return_spectra=True):
        """ Returns CMB spectra of scalar modes only and tensor modes only (with r=1)
            Returns either CMB spectra in the usual form,
            or in the red_cov form if return_spectra == False
        """
        if self.nstokes == 2:
            partial_indices_polar = np.array([1,2])
        elif self.nstokes == 1:
            partial_indices_polar = np.array([0])
        else:
            partial_indices_polar = np.arange(4)

        all_spectra_r0 = generate_power_spectra_CAMB(self.nside*2, r=0, typeless_bool=True)
        all_spectra_r1 = generate_power_spectra_CAMB(self.nside*2, r=1, typeless_bool=True)

        camb_cls_r0 = all_spectra_r0['total'][:self.lmax+1,partial_indices_polar]

        tensor_spectra_r1 = all_spectra_r1['tensor'][:self.lmax+1,partial_indices_polar]

        theoretical_r1_tensor = np.zeros((self.number_correlations,self.lmax+1))
        theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

        theoretical_r1_tensor[:self.nstokes,...] = tensor_spectra_r1.T
        theoretical_r0_total[:self.nstokes,...] = camb_cls_r0.T

        if return_spectra:
            return theoretical_r0_total, theoretical_r1_tensor

        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[self.lmin:]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)[self.lmin:]
        return theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor


    def generate_input_freq_maps_from_fgs(self, freq_maps_fgs, return_only_freq_maps=True, return_only_maps=False):
        indices_polar = np.array([1,2,4])

        theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = self.generate_CMB(return_spectra=False)

        c_ell_select_only_Bmodes = np.zeros((6,self.lmax+1))
        c_ell_select_only_Bmodes[2,self.lmin:] = 1
        red_cov_select_Bmodes = get_reduced_matrix_from_c_ell(c_ell_select_only_Bmodes[indices_polar,...])[self.lmin:,...]

        true_cmb_specra = get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total + self.r_true*theoretical_red_cov_r1_tensor)
        true_cmb_specra_extended = np.zeros((6,self.lmax+1))
        true_cmb_specra_extended[indices_polar,self.lmin:] = true_cmb_specra

        input_cmb_maps_alt = hp.synfast(true_cmb_specra_extended, nside=self.nside, new=True, lmax=self.lmax)[1:,...]

        input_cmb_maps = np.repeat(input_cmb_maps_alt.ravel(order='F'), self.number_frequencies).reshape((self.number_frequencies,self.nstokes,self.npix),order='F')
        input_freq_maps = input_cmb_maps + freq_maps_fgs

        # true_red_cov_cmb_specra = get_reduced_matrix_from_c_ell(true_cmb_specra)
        if return_only_freq_maps:
            return input_freq_maps
        
        if return_only_maps:
            return input_freq_maps, input_cmb_maps

        return input_freq_maps, input_cmb_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor

    # @property
    # def all_samples_eta(self):
    #     return jnp.empty(0)
    
    # @property
    # def all_params_mixing_matrix_samples(self):
    #     return jnp.empty(0)
    
    # @property
    # def all_samples_wiener_filter_maps(self):
    #     return jnp.empty(0)
    
    # @property
    # def all_samples_fluctuation_maps(self):
    #     return jnp.empty(0)
    
    # @property
    # def all_samples_s_c(self):
    #     return jnp.empty(0)

    # @property
    # def all_samples_r(self):
    #     return jnp.empty(0)

    # @property
    # def all_samples_CMB_c_ell(self):
    #     return jnp.empty(0)


    def update_variable(self, all_samples, new_samples_to_add):
        if jnp.size(all_samples) == 0:
            return new_samples_to_add
        elif jnp.size(new_samples_to_add.shape) == 1:
            return jnp.hstack([all_samples,new_samples_to_add])
        else:
            return jnp.vstack([all_samples,new_samples_to_add])

    def update_samples(self, all_samples):
        indice_s_c = 1
        # if self.sample_eta_B_f:
        if not(self.cheap_save):
            self.all_samples_eta = self.update_variable(self.all_samples_eta, all_samples[0])

        if not(self.very_cheap_save):
            self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, all_samples[indice_s_c])
            self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, all_samples[indice_s_c+1])
        # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])
        if self.sample_C_inv_Wishart:
            if all_samples[indice_s_c+2].shape[1] == self.lmax+1-self.lmin:
                all_samples_CMB_c_ell = jnp.array([get_c_ells_from_red_covariance_matrix(all_samples[indice_s_c+2][iteration]) for iteration in range(self.number_iterations_sampling-self.number_iterations_done)])
            else:
                all_samples_CMB_c_ell = all_samples[indice_s_c+2]
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples_CMB_c_ell)
        if self.sample_r_Metropolis:
            self.all_samples_r = self.update_variable(self.all_samples_r, all_samples[indice_s_c+3])

        self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, all_samples[5])

    def update_one_sample(self, one_sample):
        indice_s_c = 1
        # if self.sample_eta_B_f:
        self.all_samples_eta = self.update_variable(self.all_samples_eta, one_sample[0].reshape([1]+list(one_sample[0].shape)))
        self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, one_sample[5].reshape([1]+list(one_sample[5].shape)))
        # else:
        #     indice_s_c = -1

        self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, one_sample[indice_s_c].reshape([1]+list(one_sample[indice_s_c].shape)))
        self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, one_sample[indice_s_c+1].reshape([1]+list(one_sample[indice_s_c+1].shape)))
        # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])

        if one_sample[indice_s_c+2].shape[0] == self.lmax+1-self.lmin:
            one_sample_CMB_c_ell = get_c_ells_from_red_covariance_matrix(one_sample[indice_s_c+2])
        else:
            one_sample_CMB_c_ell = one_sample[indice_s_c+2]
        self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, one_sample_CMB_c_ell.reshape([1]+list(one_sample_CMB_c_ell.shape)))
        self.all_samples_r = self.update_variable(self.all_samples_r, one_sample[indice_s_c+3])
        # if self.sample_r_Metropolis:
        #     self.all_samples_r = self.update_variable(self.all_samples_r, all_samples[indice_s_c+2])
        # else:
        #     self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples[indice_s_c+2])

    # def compute_covariance_1d(self, iteration, all_samples):
    #     """ Give new covariance matrix from the samples of a 1d variable
    #         assuming after iteration the samples are put to 0
    #     """
    #     dimension_sample = 1
    #     mean_samples = all_samples.sum(axis=0)/(iteration+1)

    #     empirical_covariance = (jnp.einsum('t,t',all_samples,all_samples)
    #                             - (iteration+1)*(mean_samples**2))/(iteration)

    #     return (self.scale_param/dimension_sample)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_sample))
    

    # def compute_covariance_nd(self, iteration, all_samples):
    #     """ Give new covariance matrix from the samples of a nd variable
    #         assuming after iteration the samples are put to 0
    #     """
    #     dimension_sample = all_samples.shape[-1]
    #     mean_samples = all_samples.sum(axis=0)/(iteration+1)

    #     empirical_covariance = (jnp.einsum('ti,tj->tij',all_samples,all_samples).sum(axis=0) 
    #                             - (iteration+1)*jnp.einsum('i,j->ij',mean_samples,mean_samples))/(iteration)

    #     return (self.scale_param/dimension_sample)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_sample))

    def spectral_likelihood_dB(self, new_params_mixing_matrix, full_data, **model_kwargs):
        """
        Returns a list of the derivatives of -log(L_sp)
        per each spectral parameter
        """

        # self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        self.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)

        # new_mixing_matrix = self._fake_mixing_matrix.get_B(jax_use=True)
        # new_mixing_matrix_dB = self._fake_mixing_matrix.get_B_db(jax_use=True)
        new_mixing_matrix = self.get_B(jax_use=True)
        new_mixing_matrix_dB = self.get_B_db(jax_use=True)

        invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
        BtinvN = get_BtinvN(self.freq_inverse_noise, new_mixing_matrix, jax_use=True)
        
        BtinvNd = jnp.einsum('cf, fsp -> csp', BtinvN, full_data)

        P = jnp.einsum('ef,fc,cg,lg,lh->eh', self.freq_inverse_noise, new_mixing_matrix, invBtinvNB, new_mixing_matrix, self.freq_inverse_noise)

        logL_dB = jnp.einsum('csp,ck,bfk,fg,gsp->b', BtinvNd, invBtinvNB, new_mixing_matrix_dB, self.freq_inverse_noise-P, full_data)

        return -logL_dB

    def spectral_likelihood_dB_f(self, new_params_mixing_matrix, full_data_without_CMB, **model_kwargs):
        """
        Returns a list of the derivatives of -log(L_sp)
        per each spectral parameter
        """

        # self._fake_mixing_matrix.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)
        self.update_params(new_params_mixing_matrix.reshape((self.number_frequencies-jnp.size(self.pos_special_freqs), self.number_components-1),order='F'),jax_use=True)

        # new_mixing_matrix_fg = self._fake_mixing_matrix.get_B(jax_use=True)[:,1:]
        # new_mixing_matrix_dB_fg = self._fake_mixing_matrix.get_B_db(jax_use=True)[:,:,1:]
        new_mixing_matrix_fg = self.get_B(jax_use=True)[:,1:]
        new_mixing_matrix_dB_fg = self.get_B_db(jax_use=True)[:,:,1:]

        invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, new_mixing_matrix_fg, jax_use=True)
        BtinvN = get_BtinvN(self.freq_inverse_noise, new_mixing_matrix_fg, jax_use=True)
        
        BtinvNd = jnp.einsum('cf, fsp -> csp', BtinvN, full_data_without_CMB)

        P = jnp.einsum('ef,fc,cg,lg,lh->eh', self.freq_inverse_noise, new_mixing_matrix_fg, invBtinvNB, new_mixing_matrix_fg, self.freq_inverse_noise)

        logL_dB = jnp.einsum('csp,ck,bfk,fg,gsp->b', BtinvNd, invBtinvNB, new_mixing_matrix_dB_fg, self.freq_inverse_noise-P, full_data_without_CMB)

        return -logL_dB

    # def parameter_estimate !!!!
    # TO DO !!!!!!!!!!!!!!!!!


    def perform_sampling(self, input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=0, initial_wiener_filter_term=jnp.empty(0), initial_fluctuation_maps=jnp.empty(0),
                         theoretical_r0_total=jnp.empty(0), theoretical_r1_tensor=jnp.empty(0)):
        """ Perform sampling steps with :
                1. A CG on variable eta for (S_approx + mixed_noise) eta = S_approx^(1/2) x + E^t (B^t N^{-1} B)^{-1} E noise^(1/2) y
                2. A CG for the Wiener filter variable s_c : (s_c - s_c,ML)^t (S_c + E^t (B^t N^{-1} B)^{-1} E) (s_c - s_c,ML)
                3. The c_ell sampling assuming inverse Wishart distribution
                4. Mixing matrix B_f sampling with : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t (S_{approx} + E^t (B^T N^{-1} B)^{-1} E) eta

            Parameters
            ----------
            input_freq_maps : data of initial frequency maps, dimensions [frequencies, nstokes, npix]
            
            tot_cov_first_guess : total covariance first guess, composed of all c_ell correlations in order (for polarization [EE, BB, EB])

            depth_p: depth for the noise covariance calculation, in uK.arcmin

            c_ell_approx : 

            init_params_mixing_matrix : 
            
            pos_special_freqs :

            Returns
            -------
        """

        # Disabling all chex checks to speed up the code
        # chx acts like an assert, but is JAX compatible
        if self.disable_chex:
            print("Disabling chex !!!", flush=True)
            chx.disable_asserts()

        # Series of tests

        ## Getting only the relevant spectra
        if self.nstokes == 2:
            indices_to_consider = np.array([1,2,4])
            partial_indices_polar = indices_to_consider[:self.nstokes]
        elif self.nstokes == 1:
            indices_to_consider = np.array([0])
        else:
            indices_to_consider = np.arange(6)

        ## Testing the initial WF term, else initialize it properly
        if len(initial_wiener_filter_term) == 0:
            wiener_filter_term = jnp.zeros((self.nstokes,self.npix))
        else:
            assert len(initial_wiener_filter_term.shape) == 2
            assert initial_wiener_filter_term.shape[0] == self.nstokes
            assert initial_wiener_filter_term.shape[1] == self.npix
            wiener_filter_term = initial_wiener_filter_term
        
        ## Testing the initial fluctuation term, else initialize it properly
        if len(initial_fluctuation_maps) == 0:
            fluctuation_maps = jnp.zeros((self.nstokes,self.npix))
        else:
            assert len(initial_fluctuation_maps.shape) == 2
            assert initial_fluctuation_maps.shape[0] == self.nstokes
            assert initial_fluctuation_maps.shape[1] == self.npix
            fluctuation_maps = initial_fluctuation_maps

        ## Testing the initial spectra given in case the sampling is done with r
        if self.sample_r_Metropolis:
            assert len(theoretical_r0_total.shape) == 2
            assert (theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin) #or (theoretical_r0_total.shape[1] == self.lmax + 1)
            assert len(theoretical_r1_tensor.shape) == 2
            assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

            theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
            theoretical_red_cov_r1_tensor  = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)

        ## Testing the initial CMB spectra given
        if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):    
            CMB_c_ell = CMB_c_ell[indices_to_consider,:]
            c_ell_approx = c_ell_approx[indices_to_consider,:]
        
        ## Final set of tests
        assert len(CMB_c_ell.shape) == 2
        assert CMB_c_ell.shape[1] == self.lmax + 1
        assert len(c_ell_approx.shape) == 2
        assert c_ell_approx.shape[1] == self.lmax + 1


        # Preparing for the full Gibbs sampling
        len_pos_special_freqs = len(self.pos_special_freqs)

        ## Initial guesses
        initial_eta = jnp.zeros((self.nstokes,self.npix))
        params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix).reshape(
                                            ((self.number_frequencies-len_pos_special_freqs),self.number_correlations-1), order='F')

        ## CMB covariance preparation in the format [lmax,nstokes,nstokes]
        red_cov_approx_matrix = jnp.array(get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin:,...])
        red_cov_matrix = get_reduced_matrix_from_c_ell(CMB_c_ell)[self.lmin:,...]

        ## Preparation of the mixing matrix
        self.mixing_matrix_obj = MixingMatrix(self.frequency_array, self.number_components, params_mixing_matrix_init_sample, pos_special_freqs=self.pos_special_freqs)


        ## Jitting the sampling function
        jitted_sample_eta = jax.jit(self.get_sampling_eta_v2, static_argnames=['suppress_low_modes'])
        
        func_fixed_covariance_eta = self.get_conditional_proba_correction_likelihood_JAX_v2c

        # jitted_get_fluctuating_term = jax.jit(self.get_fluctuating_term_maps)
        # jitted_solve_wiener_filter_term = jax.jit(self.solve_generalized_wiener_filter_term)

        # jitted_get_fluctuating_term = jax.jit(self.get_fluctuating_term_maps_v2)
        # jitted_solve_wiener_filter_term = jax.jit(self.solve_generalized_wiener_filter_term_v2)

        sampling_func_WF = self.solve_generalized_wiener_filter_term_v2c
        sampling_func_Fluct = self.get_fluctuating_term_maps_v2c
        # if self.use_old_s_c_sampling:
        #     sampling_func_WF = self.solve_generalized_wiener_filter_term
        #     sampling_func_Fluct = self.get_fluctuating_term_maps


        # jitted_get_inverse_wishart_sampling_from_c_ells = jax.jit(self.get_inverse_wishart_sampling_from_c_ells, static_argnames=['q_prior', 'option_ell_2', 'tol'])
        # jitted_get_inverse_wishart_sampling_from_c_ells = jax.jit(self.get_inverse_wishart_sampling_from_c_ells)
        # jitted_get_inverse_wishart_sampling_from_c_ells = jax.jit(self.get_inverse_gamma_sampling_from_c_ells) # Use of gamma distribution instead of inverse Wishart
        func_get_inverse_wishart_sampling_from_c_ells = self.get_inverse_wishart_sampling_from_c_ells
        if self.use_binning:
            func_get_inverse_wishart_sampling_from_c_ells = self.get_binned_inverse_wishart_sampling_from_c_ells_v2
        # func_get_inverse_wishart_sampling_from_c_ells = self.get_inverse_gamma_sampling_from_c_ells

        if self.lognormal_r:
            # Using lognormal proposal distribution for r
            r_sampling_MH = single_lognormal_Metropolis_Hasting_step
        else:
            # Using normal proposal distribution for r
            r_sampling_MH = single_Metropolis_Hasting_step
        
        jitted_single_Metropolis_Hasting_step_r = jax.jit(single_Metropolis_Hasting_step, static_argnames=['log_proba'])

        jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v2b_JAX)
        sampling_func = separate_single_MH_step_index_accelerated

        if self.biased_version or self.perturbation_eta_covariance:
            print("Using biased version or perturbation version of mixing matrix sampling !!!", flush=True)
            # jitted_Bf_func_sampling = jax.jit(self.get_biased_conditional_proba_mixing_matrix_v2_JAX)
            jitted_Bf_func_sampling = jax.jit(self.get_conditional_proba_mixing_matrix_v3_JAX)
            sampling_func = separate_single_MH_step_index


        ## Preparing the scalar quantities
        PRNGKey = random.PRNGKey(self.seed)

        actual_number_of_iterations = self.number_iterations_sampling - self.number_iterations_done

        dimension_param_B_f = (self.number_frequencies-len_pos_special_freqs)*(self.number_correlations-1)
        number_correlations = self.number_correlations
        red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)


        ## Preparing the step-size for Metropolis-within-Gibbs of B_f sampling
        try :
            initial_step_size_Bf = jnp.array(jnp.diag(jsp.linalg.sqrtm(self.covariance_step_size_B_f)), dtype=jnp.float64)
        except:
            initial_step_size_Bf = jnp.array(jnp.diag(jnp.sqrt(self.covariance_step_size_B_f)), dtype=jnp.float64)
        print('Step-size B_f', initial_step_size_Bf, flush=True)

        ## Few prints to re-check the toml parameters chosen
        if not(self.sample_eta_B_f):
            print("Not sampling for eta and B_f, only for s_c and the covariance !", flush=True)
        if self.sample_r_Metropolis:
            print("Sample for r instead of C !", flush=True)
        else:
            print("Sample for C with inverse Wishart !", flush=True)

        print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done", flush=True)

        @scan_tqdm(actual_number_of_iterations, )
        def all_sampling_steps(carry, iteration):
            """ Gibbs sampling function, performing the following steps :
                - Sampling of eta, for the C_approx term
                - Sampling of s_c, for the constrained CMB map realization
                - Sampling of C, for the CMB covariance matrix
                - Sampling of B_f, for the mixing matrix

                :param carry: tuple containing the following variables at 1 iteration : eta maps, WF maps, fluctuation maps, CMB covariance, r samples, B_f samples, PRNGKey, s_c samples
                :param iteration: current iteration number

                :return: tuple containing the following variables at the last iteration : eta maps, WF maps, fluctuation maps, CMB covariance, r sample, B_f sample, PRNGKey, s_c sample
            """

            # Extracting the variables from the carry
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, _all_r_samples, _all_B_f_samples, PRNGKey, old_s_c_sample, inverse_term = carry
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, _all_r_samples, params_mixing_matrix_sample, PRNGKey, old_s_c_sample, inverse_term = carry
            eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample, PRNGKey, old_s_c_sample, inverse_term = carry

            # Preparing a new PRNGKey for eta sampling
            PRNGKey, subPRNGKey = random.split(PRNGKey)

            # Extracting the mixing matrix parameters and initializing the new one
            params_mixing_matrix_sample = params_mixing_matrix_sample.reshape((self.number_frequencies-len_pos_special_freqs,number_correlations-1),order='F')
            self.mixing_matrix_obj.update_params(params_mixing_matrix_sample)
            mixing_matrix_sampled = self.mixing_matrix_obj.get_B(jax_use=True)

            # Few checks for the mixing matrix
            # chx.assert_axis_dimension(mixing_matrix_sampled, 0, self.number_frequencies)
            # chx.assert_axis_dimension(mixing_matrix_sampled, 1, self.number_components)
            chx.assert_shape(mixing_matrix_sampled, (self.number_frequencies, self.number_components))

            # Application of new mixing matrix to the noise covariance and extracted CMB map from data
            invBtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, mixing_matrix_sampled, jax_use=True)
            BtinvN_sqrt = get_BtinvN(jnp.sqrt(self.freq_inverse_noise), mixing_matrix_sampled, jax_use=True)
            s_cML = get_Wd(self.freq_inverse_noise, mixing_matrix_sampled, input_freq_maps, jax_use=True)[0]

            # Sampling step 1 : sampling of Gaussian variable eta
            if self.sample_eta_B_f and not(self.biased_version):
                # Preparing random variables
                # map_random_x = jnp.empty(0)
                # map_random_y = jnp.empty(0)
                map_random_x = None
                map_random_y = None
                
                # Sampling eta maps
                new_eta_maps_sample = self.get_sampling_eta_v2(red_cov_approx_matrix, invBtinvNB, BtinvN_sqrt, 
                                                               subPRNGKey, map_random_x=map_random_x, map_random_y=map_random_y, 
                                                               suppress_low_modes=True)
                # new_eta_maps_sample = jitted_sample_eta(red_cov_approx_matrix, invBtinvNB, BtinvN_sqrt, 
                #                                                subPRNGKey, map_random_x=map_random_x, map_random_y=map_random_y, 
                #                                                suppress_low_modes=True)

                eta_maps_sample = new_eta_maps_sample
                
                # Checking shape of the resulting maps
                chx.assert_shape(eta_maps_sample, (self.nstokes, self.npix))

                # Computing the associated log proba term fixed correction covariance for the B_f sampling
                # if self.fixed_eta_covariance:
                #     log_proba_perturbation, inverse_term = func_fixed_covariance_eta(mixing_matrix_sampled, eta_maps_sample, red_cov_approx_matrix, previous_inverse=inverse_term, return_inverse=True)
                # else:
                #     log_proba_perturbation = None
                
                if self.perturbation_eta_covariance:
                    _, inverse_term = func_fixed_covariance_eta(mixing_matrix_sampled, eta_maps_sample, red_cov_approx_matrix, previous_inverse=inverse_term, return_inverse=True)
                

            # Sampling step 2 : sampling of Gaussian variable s_c, contrained CMB map realization

            ## Geting the Wiener filter term, mean of the variable s_c
            red_cov_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_matrix_sample)
            # wiener_filter_term = self.solve_generalized_wiener_filter_term(s_cML, red_cov_matrix_sample, invBtinvNB, initial_guess=jnp.copy(WF_term_maps))
            # wiener_filter_term = self.solve_generalized_wiener_filter_term_v2(s_cML, red_cov_matrix_sample, invBtinvNB, initial_guess=jnp.copy(WF_term_maps))
            # wiener_filter_term = self.solve_generalized_wiener_filter_term_v2c(s_cML, red_cov_matrix_sample, invBtinvNB, initial_guess=jnp.copy(WF_term_maps))
            # wiener_filter_term = jitted_solve_wiener_filter_term(s_cML, red_cov_matrix_sample, invBtinvNB, initial_guess=jnp.copy(WF_term_maps))
            initial_guess_WF = maps_x_red_covariance_cell_JAX(WF_term_maps, jnp.linalg.pinv(red_cov_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
            wiener_filter_term = sampling_func_WF(s_cML, red_cov_matrix_sqrt, invBtinvNB, initial_guess=initial_guess_WF)

            ## Preparing the random variables for the fluctuation term
            PRNGKey, new_subPRNGKey = random.split(PRNGKey)
            # map_random_realization_xi = jnp.empty(0)
            # map_random_realization_chi = jnp.empty(0)
            map_random_realization_xi = None
            map_random_realization_chi = None

            ## Getting the fluctuation maps terms, for the variance of the variable s_c
            # fluctuation_maps = self.get_fluctuating_term_maps_v2(red_cov_matrix_sample, invBtinvNB, BtinvN_sqrt, new_subPRNGKey, 
            #                                                   map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, 
            #                                                   initial_guess=jnp.copy(fluct_maps))
            # fluctuation_maps = self.get_fluctuating_term_maps_v2c(red_cov_matrix_sample, invBtinvNB, BtinvN_sqrt, new_subPRNGKey, 
            #                                                   map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, 
            #                                                   initial_guess=jnp.copy(fluct_maps))
            # fluctuation_maps = jitted_get_fluctuating_term(red_cov_matrix_sample, invBtinvNB, BtinvN_sqrt, new_subPRNGKey, 
            #                                                   map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, 
            #                                                   initial_guess=jnp.copy(fluct_maps))
            initial_guess_Fluct = maps_x_red_covariance_cell_JAX(fluct_maps, jnp.linalg.pinv(red_cov_matrix_sqrt), nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)
            fluctuation_maps = sampling_func_Fluct(red_cov_matrix_sqrt, invBtinvNB, BtinvN_sqrt, new_subPRNGKey, 
                                                              map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, 
                                                              initial_guess=initial_guess_Fluct)

            s_c_sample = fluctuation_maps + wiener_filter_term

            ## Checking the shape of the resulting maps
            chx.assert_shape(wiener_filter_term, (self.nstokes, self.npix))
            chx.assert_shape(fluctuation_maps, (self.nstokes, self.npix))
            chx.assert_shape(s_c_sample, (self.nstokes, self.npix))


            # Sampling step 3 : sampling of CMB covariance C
            
            ## Preparing the c_ell which will be used for the sampling
            c_ells_Wishart_ = get_cell_from_map_jax(s_c_sample, lmax=self.lmax, n_iter=self.n_iter)
            # c_ells_Wishart_modified = c_ells_Wishart_*(2*jnp.arange(self.lmax+1) + 1)
            ### Getting them in the format [lmax,nstokes,nstokes]
            # red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)[self.lmin:]
            red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_*(2*jnp.arange(self.lmax+1) + 1))[self.lmin:]

            ## Preparing the new PRNGkey
            PRNGKey, new_subPRNGKey_2 = random.split(PRNGKey)

            ## Performing the sampling
            if self.sample_C_inv_Wishart:
                # Sampling C with inverse Wishart
                # red_cov_matrix_sample = jitted_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_modified, PRNGKey=new_subPRNGKey_2)
                # red_cov_matrix_sample = func_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_modified, PRNGKey=new_subPRNGKey_2)
                # red_cov_matrix_sample = func_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_, PRNGKey=new_subPRNGKey_2)
                # red_cov_matrix_sample = func_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_[:,self.lmin:], PRNGKey=new_subPRNGKey_2)
                red_cov_matrix_sample = func_get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart_[:,self.lmin:], PRNGKey=new_subPRNGKey_2, 
                                                                    old_sample=red_cov_matrix_sample, acceptance_posdef=self.acceptance_posdef)

            elif self.sample_r_Metropolis:
                # Sampling r which will parametrize C(r) = C_scalar + r*C_tensor

                # Possibly use automatic step size
                # if self.use_automatic_step_size:
                #     step_size_r = jnp.where(iteration<self.num_sample_AM,  self.step_size_r, jnp.sqrt(self.compute_covariance_1d(iteration, _all_r_samples)))
                # else:
                #     step_size_r = self.step_size_r

                # r_sample = r_sampling_MH(random_PRNGKey=new_subPRNGKey_2, old_sample=_all_r_samples[iteration],
                #                                           step_size=step_size_r, log_proba=self.get_conditional_proba_C_from_r, 
                #                                           red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                # _all_r_samples = _all_r_samples.at[iteration+1].set(r_sample)

                r_sample = r_sampling_MH(random_PRNGKey=new_subPRNGKey_2, old_sample=r_sample,
                                                          step_size=self.step_size_r, log_proba=self.get_conditional_proba_C_from_r, 
                                                          red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                # _all_r_samples = _all_r_samples.at[iteration+1].set(r_sample)

                red_cov_matrix_sample = theoretical_red_cov_r0_total + r_sample*theoretical_red_cov_r1_tensor
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')
            
            ## Checking the shape of the resulting covariance matrix, and correcting it if needed
            if red_cov_matrix_sample.shape[0] == self.lmax + 1:
                red_cov_matrix_sample = red_cov_matrix_sample[self.lmin:]            
            chx.assert_shape(red_cov_matrix_sample, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))


            # Sampling step 4 : sampling of mixing matrix B_f

            ## Preparation of sampling step 4
            # extended_CMB_maps = jnp.zeros((self.number_components, self.nstokes, self.npix))
            # extended_CMB_maps = extended_CMB_maps.at[0].set(s_c_sample)
            # full_data_without_CMB = input_freq_maps - jnp.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
            full_data_without_CMB = input_freq_maps - jnp.einsum('f,sp->fsp',mixing_matrix_sampled[:,0], s_c_sample)
            chx.assert_shape(full_data_without_CMB, (self.number_frequencies, self.nstokes, self.npix))

            ## Preparing the new PRNGKey
            PRNGKey, new_subPRNGKey_3 = random.split(PRNGKey)
            
            ## Performing the sampling
            if self.sample_eta_B_f:
                # Preparing the step-size
                step_size_Bf = initial_step_size_Bf
                # if self.use_automatic_step_size:
                #     covariance_matrix_B_f = jnp.where(iteration < self.num_sample_AM, 
                #                                     self.covariance_step_size_B_f,
                #                                     self.compute_covariance_nd(iteration, _all_B_f_samples))
                #     step_size_Bf = jnp.array(jnp.diag(jsp.linalg.sqrtm(covariance_matrix_B_f)), dtype=jnp.float64)

                # Sampling B_f

                if self.perturbation_eta_covariance or self.biased_version:
                    inverse_term_x_Capprox_root = maps_x_red_covariance_cell_JAX(inverse_term.reshape(self.nstokes,self.npix), red_cov_approx_matrix_sqrt, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter).ravel()
                    new_subPRNGKey_3, params_mixing_matrix_sample = sampling_func(random_PRNGKey=new_subPRNGKey_3, old_sample=params_mixing_matrix_sample, 
                                                            step_size=step_size_Bf, indexes_Bf=self.indexes_free_Bf,
                                                            log_proba=jitted_Bf_func_sampling,
                                                            full_data_without_CMB=full_data_without_CMB, component_eta_maps=eta_maps_sample, 
                                                            red_cov_approx_matrix=red_cov_approx_matrix, previous_inverse=inverse_term,
                                                            previous_inverse_x_Capprox_root=inverse_term_x_Capprox_root,
                                                            old_params_mixing_matrix=params_mixing_matrix_sample,
                                                            biased_bool=self.biased_version)
                else:
                    new_subPRNGKey_3, params_mixing_matrix_sample, inverse_term = sampling_func(random_PRNGKey=new_subPRNGKey_3, old_sample=params_mixing_matrix_sample, 
                                                            step_size=step_size_Bf, indexes_Bf=self.indexes_free_Bf,
                                                            log_proba=jitted_Bf_func_sampling,
                                                            full_data_without_CMB=full_data_without_CMB, component_eta_maps=eta_maps_sample, 
                                                            red_cov_approx_matrix=red_cov_approx_matrix,
                                                            previous_inverse=inverse_term,
                                                            biased_bool=self.biased_version)

                # Checking the shape of the resulting mixing matrix
                chx.assert_axis_dimension(params_mixing_matrix_sample, 0, self.number_frequencies-len_pos_special_freqs)
                params_mixing_matrix_sample = params_mixing_matrix_sample.reshape((self.number_frequencies-len_pos_special_freqs,number_correlations-1),order='F')


            new_carry = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample, PRNGKey, s_c_sample, inverse_term)
            all_samples = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample)

            return new_carry, all_samples

        # Initializing r and B_f samples

        initial_carry = (initial_eta, 
                         wiener_filter_term, fluctuation_maps, 
                         red_cov_matrix,
                         initial_guess_r,
                         params_mixing_matrix_init_sample,
                         PRNGKey,
                         wiener_filter_term+fluctuation_maps,
                         jnp.zeros_like(initial_eta))
        initial_carry_0 = (initial_eta, 
                         wiener_filter_term, fluctuation_maps, 
                         red_cov_matrix,
                         initial_guess_r,
                         params_mixing_matrix_init_sample,
                         PRNGKey)
        self.update_one_sample(initial_carry_0)

        jitted_all_sampling_steps = jax.jit(all_sampling_steps)

        # def my_scan(f, init, xs, length=None):
        #     if xs is None:
        #         xs = [None] * length
        #     carry = init
        #     ys = []
        #     for x in xs:
        #         time_iter = time.time()
        #         carry, y = f(carry, x)
        #         ys.append(y)
        #         print("##### Iteration in {} seconds".format(time.time()-time_iter), flush=True)
        #     return carry, jnp.stack(ys)
        # jitted_scan = jax.jit(jlx.scan)
        time_start_sampling = time.time()
        # Start sampling !!!
        # last_sample, all_samples = jlx.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        
        # last_sample, all_samples = jlx.scan(jitted_all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        last_sample, all_samples = jlx.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))

        # last_sample, all_samples = my_scan(all_sampling_steps, initial_carry, None, length=actual_number_of_iterations)
        time_full_chain = (time.time()-time_start_sampling)/60      
        print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

        # Saving the samples as attributes of the Sampler object
        self.update_samples(all_samples)
        self.number_iterations_done = self.number_iterations_sampling

    def continue_sampling(self,input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=0, initial_wiener_filter_term=jnp.empty(0), initial_fluctuation_maps=jnp.empty(0),
                         theoretical_r0_total=jnp.empty(0), theoretical_r1_tensor=jnp.empty(0)):

        if self.sample_r_Metropolis:
            CMB_c_ell = theoretical_r0_total + initial_guess_r*theoretical_r1_tensor

        self.perform_sampling(self, input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=initial_guess_r, initial_wiener_filter_term=initial_wiener_filter_term, initial_fluctuation_maps=initial_fluctuation_maps,
                         theoretical_r0_total=theoretical_r0_total, theoretical_r1_tensor=theoretical_r1_tensor)


def create_MICMAC_sampler_from_toml_file(path_toml_file):
    """ Create a MICMAC_Sampler object from the path of a toml file 
    """
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    if dictionary_parameters['instrument_name'] != 'customized_instrument':
        instrument = fgbuster.get_instrument(dictionary_parameters['instrument_name'])
        dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
        dictionary_parameters['freq_inverse_noise'] = get_noise_covar(instrument['depth_p'], dictionary_parameters['nside'])

    # del dictionary_parameters['instrument_name']
    return MICMAC_Sampler(**dictionary_parameters)
