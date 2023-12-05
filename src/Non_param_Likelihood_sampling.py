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
                 n_iter=8, limit_iter_cg=2000, tolerance_CG=10**(-12),
                 num_sample_AM = 1000, epsilon_cov = 10**(-20), scale_param = 2.38**2,

                 cheap_save=True, very_cheap_save=False,
                 biased_version=False,
                 r_true=0, only_select_Bmodes=False, no_Emodes_CMB=False, 
                 sample_eta_B_f=True, harmonic_correction=False,
                 sample_r_Metropolis=True, sample_C_inv_Wishart=False,
                 n_walkers_Metropolis=1, step_size_B_f=10**(-4), step_size_r=10**(-4),
                 fullsky_ver=True, slow_ver=False,

                 progress_bar=False,
                 number_iterations_sampling=100, number_iterations_done=0, seed=0,
                 disable_chex=True):
        """ Non parametric likelihood sampling object
        """

        super(MICMAC_Sampler,self).__init__(nside=nside,lmax=lmax,nstokes=nstokes,lmin=lmin,
                                            frequency_array=frequency_array,freq_inverse_noise=freq_inverse_noise, 
                                            pos_special_freqs=pos_special_freqs,number_components=number_components,
                                            n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance_CG=tolerance_CG)

        # Quick test parameters
        self.fullsky_ver = bool(fullsky_ver)
        self.slow_ver = bool(slow_ver)
        self.sample_eta_B_f = bool(sample_eta_B_f)
        self.biased_version = bool(biased_version)
        self.harmonic_correction = bool(harmonic_correction)
        self.cheap_save = bool(cheap_save)
        self.progress_bar = progress_bar
        self.disable_chex = disable_chex

        # CMB parameters
        self.r_true = float(r_true)
        self.only_select_Bmodes = bool(only_select_Bmodes)
        self.no_Emodes_CMB = bool(no_Emodes_CMB)
        assert (sample_r_Metropolis or sample_C_inv_Wishart == True) and (sample_r_Metropolis and sample_C_inv_Wishart == False)
        self.sample_r_Metropolis = bool(sample_r_Metropolis)
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)


        # Metropolis-Hastings parameters
        self.n_walkers_Metropolis = int(n_walkers_Metropolis) # Number of walkers for the MCMC to sample the mixing matrix or r
        # self.step_size_B_f = step_size_B_f
        self.covariance_step_size_B_f = jnp.diag((jnp.ravel(step_size_B_f,order='F')**2)*jnp.ones((self.number_frequencies-len(pos_special_freqs))*(self.number_correlations-1)))
        self.step_size_r = step_size_r
        # self.number_steps_sampler_B_f = int(number_steps_sampler_B_f) # Maximum number of steps for the Metropolis-Hasting to sample the mixing matrix
        # self.number_steps_sampler_r = int(number_steps_sampler_r)
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
        # lensing_spectra_r0 = all_spectra_r0['lensed_scalar'][:self.lmax+1,partial_indices_polar]
        # tensor_spectra_r0 = all_spectra_r0['tensor'][:self.lmax+1,partial_indices_polar]
        # lens_potential_spectra_r0 = all_spectra_r0['lens_potential'][:self.lmax+1,partial_indices_polar]
        # unlensed_scalar_spectra_r0 = all_spectra_r0['unlensed_scalar'][:self.lmax+1,partial_indices_polar]
        # unlensed_total_spectra_r0 = all_spectra_r0['unlensed_total'][:self.lmax+1,partial_indices_polar]

        # camb_cls_r1 = all_spectra_r1['total'][:self.lmax+1,partial_indices_polar]
        # lensing_spectra_r1 = all_spectra_r1['lensed_scalar'][:self.lmax+1,partial_indices_polar]
        tensor_spectra_r1 = all_spectra_r1['tensor'][:self.lmax+1,partial_indices_polar]
        # lens_potential_spectra_r1 = all_spectra_r1['lens_potential'][:self.lmax+1,partial_indices_polar]
        # unlensed_scalar_spectra_r1 = all_spectra_r1['unlensed_scalar'][:self.lmax+1,partial_indices_polar]
        # unlensed_total_spectra_r1 = all_spectra_r1['unlensed_total'][:self.lmax+1,partial_indices_polar]

        theoretical_r1_tensor = np.zeros((self.number_correlations,self.lmax+1))
        theoretical_r0_total = np.zeros_like(theoretical_r1_tensor)

        theoretical_r1_tensor[:self.nstokes,...] = tensor_spectra_r1.T
        # theoretical_r0_total[:self.nstokes,...] = unlensed_scalar_spectra_r0.T
        theoretical_r0_total[:self.nstokes,...] = camb_cls_r0.T

        if return_spectra:
            return theoretical_r0_total, theoretical_r1_tensor

        theoretical_red_cov_r1_tensor = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)[self.lmin:]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)[self.lmin:]
        return theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor


    def generate_input_freq_maps_from_fgs(self, freq_maps_fgs, return_only_freq_maps=True):
        indices_polar = np.array([1,2,4])

        theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor = self.generate_CMB(return_spectra=False)

        c_ell_select_only_Bmodes = np.zeros((6,self.lmax+1))
        c_ell_select_only_Bmodes[2,self.lmin:] = 1
        red_cov_select_Bmodes = get_reduced_matrix_from_c_ell(c_ell_select_only_Bmodes[indices_polar,...])[self.lmin:,...]

        if self.only_select_Bmodes:
            theoretical_red_cov_r1_tensor = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, theoretical_red_cov_r1_tensor)
            theoretical_red_cov_r0_total = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, theoretical_red_cov_r0_total)
            # theoretical_red_cov_r1_tensor[:,:,0] = 0
            # theoretical_red_cov_r0_total[:,:,0] = 0
            theoretical_red_cov_r1_tensor[:,0,0] = 10**(-30)
            theoretical_red_cov_r0_total[:,0,0] = 10**(-30)

            # theoretical_r0_total[0,:] = 10**(-30)
            # theoretical_r1_tensor[0,:] = 10**(-30)
            # theoretical_r0_total[2,:] = 0
            # theoretical_r1_tensor[2,:] = 0
            for freq in range(self.number_frequencies):
                    freq_maps_fgs[freq] = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(freq_maps_fgs[freq]), red_cov_select_Bmodes, lmin=self.lmin, n_iter=self.n_iter)

        if self.no_Emodes_CMB:
            theoretical_red_cov_r0_total = np.einsum('lkj,ljm->lkm', red_cov_select_Bmodes, theoretical_red_cov_r0_total)
            # theoretical_red_cov_r0_total[:,:,0] = 0
            theoretical_red_cov_r0_total[:,0,0] = 10**(-30)

            # theoretical_r0_total[0,:] = 10**(-30)
            # theoretical_r0_total[2,:] = 0

        true_cmb_specra = get_c_ells_from_red_covariance_matrix(theoretical_red_cov_r0_total + self.r_true*theoretical_red_cov_r1_tensor)
        true_cmb_specra_extended = np.zeros((6,self.lmax+1))
        true_cmb_specra_extended[indices_polar,self.lmin:] = true_cmb_specra

        input_cmb_maps_alt = hp.synfast(true_cmb_specra_extended, nside=self.nside, new=True, lmax=self.lmax)[1:,...]

        input_cmb_maps = np.repeat(input_cmb_maps_alt.ravel(order='F'), self.number_frequencies).reshape((self.number_frequencies,self.nstokes,self.npix),order='F')
        input_freq_maps = input_cmb_maps + freq_maps_fgs

        # true_red_cov_cmb_specra = get_reduced_matrix_from_c_ell(true_cmb_specra)
        if return_only_freq_maps:
            return input_freq_maps

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

    @property 
    def last_sample(self):
        last_sample_dict = dict()

        # if self.sample_eta_B_f:
        last_sample_dict['eta'] = self.all_samples_eta[-1]
        last_sample_dict['param_mixing_matrix'] = self.all_params_mixing_matrix_samples[-1]


        last_sample_dict['WF_maps'] = self.all_samples_wiener_filter_maps[-1]
        last_sample_dict['Fluctuation_maps'] = self.all_samples_fluctuation_maps[-1]
        last_sample_dict['s_c_maps'] = self.all_samples_s_c[-1]

        last_sample_dict['r'] = self.all_samples_r[-1]
        last_sample_dict['CMB_c_ell'] = self.all_samples_CMB_c_ell[-1]

        # if self.sample_r_Metropolis:
        #     last_sample_dict['r'] = self.all_samples_r[-1]
        # else:
        #     last_sample_dict['CMB_c_ell'] = self.all_samples_CMB_c_ell[-1]

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
            if all_samples[indice_s_c+2].shape[1] == self.lmax+1-self.lmin:
                all_samples_CMB_c_ell = jnp.array([get_c_ells_from_red_covariance_matrix(all_samples[indice_s_c+2][iteration]) for iteration in range(self.number_iterations_sampling-self.number_iterations_done)])
            else:
                all_samples_CMB_c_ell = all_samples[indice_s_c+2]
            self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, all_samples_CMB_c_ell)
        
        if not(self.very_cheap_save):
            self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, all_samples[indice_s_c])
            self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, all_samples[indice_s_c+1])
        # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])

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

    def get_new_covariance_1d(self, iteration, all_samples):
        """ Give new covariance matrix from the samples of a 1d variable
            assuming after iteration the samples are put to 0
        """
        dimension_sample = 1
        mean_samples = all_samples.sum(axis=0)/(iteration+1)

        empirical_covariance = (jnp.einsum('t,t',all_samples,all_samples)
                                - (iteration+1)*(mean_samples**2))/(iteration)

        return (self.scale_param/dimension_sample)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_sample))
    

    def get_new_covariance_nd(self, iteration, all_samples):
        """ Give new covariance matrix from the samples of a nd variable
            assuming after iteration the samples are put to 0
        """
        dimension_sample = all_samples.shape[-1]
        mean_samples = all_samples.sum(axis=0)/(iteration+1)

        empirical_covariance = (jnp.einsum('ti,tj->tij',all_samples,all_samples).sum(axis=0) 
                                - (iteration+1)*jnp.einsum('i,j->ij',mean_samples,mean_samples))/(iteration)

        return (self.scale_param/dimension_sample)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_sample))

    # def parameter_estimate !!!!
    # TO DO !!!!!!!!!!!!!!!!!


    def perform_sampling(self, input_freq_maps, c_ell_approx, CMB_c_ell, init_params_mixing_matrix, 
                         initial_guess_r=10**(-5), initial_wiener_filter_term=jnp.empty(0), initial_fluctuation_maps=jnp.empty(0),
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

        if self.disable_chex:
            print("Disabling chex !!!")
            chx.disable_asserts()

        len_pos_special_freqs = len(self.pos_special_freqs)

        actual_number_of_iterations = self.number_iterations_sampling - self.number_iterations_done
        # all_eta = np.zeros((actual_number_of_iterations, self.nstokes, self.npix))
        # all_s_c_WF_maps = np.zeros((actual_number_of_iterations, self.nstokes, self.npix))
        # all_s_c_fluct_maps = np.zeros((actual_number_of_iterations, self.nstokes, self.npix))
        # all_cell_samples = np.zeros((actual_number_of_iterations, self.number_correlations, self.lmax+1))
        # all_r_samples = np.zeros(actual_number_of_iterations)
        # all_params_mixing_matrix_samples = np.zeros((actual_number_of_iterations, self.number_frequencies-len_pos_special_freqs, self.number_correlations-1))

         # Getting only the relevant spectra
        if self.nstokes == 2:
            indices_to_consider = np.array([1,2,4])
            partial_indices_polar = indices_to_consider[:self.nstokes]
        elif self.nstokes == 1:
            indices_to_consider = np.array([0])
        else:
            indices_to_consider = np.arange(6)
        
        
        if self.only_select_Bmodes:
            c_ell_select_only_Bmodes = np.zeros((6,self.lmax+1))
            c_ell_select_only_Bmodes[2,self.lmin:] = 1
            red_cov_select_Bmodes = get_reduced_matrix_from_c_ell(c_ell_select_only_Bmodes[indices_to_consider,...])[self.lmin:,...]

        if len(initial_wiener_filter_term) == 0:
            wiener_filter_term = jnp.zeros((self.nstokes,self.npix))
        else:
            assert len(initial_wiener_filter_term.shape) == 2
            assert initial_wiener_filter_term.shape[0] == self.nstokes
            assert initial_wiener_filter_term.shape[1] == self.npix
            wiener_filter_term = initial_wiener_filter_term
        
        if len(initial_fluctuation_maps) == 0:
            fluctuation_maps = jnp.zeros((self.nstokes,self.npix))
        else:
            assert len(initial_fluctuation_maps.shape) == 2
            assert initial_fluctuation_maps.shape[0] == self.nstokes
            assert initial_fluctuation_maps.shape[1] == self.npix
            fluctuation_maps = initial_fluctuation_maps
        s_c_sample = wiener_filter_term + fluctuation_maps

        if self.sample_r_Metropolis:
            assert len(theoretical_r0_total.shape) == 2
            assert (theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin) #or (theoretical_r0_total.shape[1] == self.lmax + 1)
            assert len(theoretical_r1_tensor.shape) == 2
            assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

            theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
            theoretical_red_cov_r1_tensor  = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)

       

        if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):    
            CMB_c_ell = CMB_c_ell[indices_to_consider,:]
            c_ell_approx = c_ell_approx[indices_to_consider,:]
        
        assert len(CMB_c_ell.shape) == 2
        assert CMB_c_ell.shape[1] == self.lmax + 1
        assert len(c_ell_approx.shape) == 2
        assert c_ell_approx.shape[1] == self.lmax + 1

        # Initial guesses
        initial_eta = jnp.zeros((self.nstokes,self.npix))
        # all_eta[0,...] = initial_eta
        # all_s_c_WF_maps[0,...] = wiener_filter_term
        # all_s_c_fluct_maps[0,...] = fluctuation_maps
        # all_cell_samples[0,...] = np.copy(CMB_c_ell)
        params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix).reshape(
                                            ((self.number_frequencies-len_pos_special_freqs),self.number_correlations-1), order='F')
        # all_params_mixing_matrix_samples[0,...] = params_mixing_matrix_init_sample
        # all_r_samples[0,...] = initial_guess_r

        
        # CMB covariance preparation
        red_cov_approx_matrix = jnp.array(get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin:,...])
        red_cov_matrix = get_reduced_matrix_from_c_ell(CMB_c_ell)[self.lmin:,...]

        self.mixing_matrix_obj = MixingMatrix(self.frequency_array, self.number_components, params_mixing_matrix_init_sample, pos_special_freqs=self.pos_special_freqs)
        # mixing_matrix_sampled = mixing_matrix_obj.get_B()


        num_warmup = 0

        # kernel_log_proba_Bf = MetropolisHastings_log(self.new_get_conditional_proba_full_likelihood_JAX_from_params, step_size=jnp.ravel(self.step_size_B_f,order='F'))
        # mcmc_kernel_log_proba_Bf = numpyro.infer.MCMC(kernel_log_proba_Bf, num_chains=self.n_walkers_Metropolis, num_warmup=num_warmup, num_samples=self.number_steps_sampler_B_f, progress_bar=self.progress_bar)

        # kernel_log_proba_r = MetropolisHastings_log(self.get_conditional_proba_C_from_r, step_size=self.step_size_r)
        # mcmc_kernel_log_proba_r = numpyro.infer.MCMC(kernel_log_proba_r, num_chains=self.n_walkers_Metropolis, num_warmup=num_warmup, num_samples=self.number_steps_sampler_r, progress_bar=self.progress_bar)

        if not(self.sample_eta_B_f):
            print("Not sampling for eta and B_f, only for s_c and the covariance !", flush=True)
        if self.sample_r_Metropolis:
            print("Sample for r instead of C !", flush=True)
        else:
            print("Sample for C with inverse Wishart !", flush=True)
        print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done", flush=True)
        # print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done with {self.overrelaxation_param} overrelaxation parameter", flush=True)
        
        PRNGKey = random.PRNGKey(self.seed)
        jitted_sample_eta = jax.jit(self.get_sampling_eta_v2, static_argnames=['suppress_low_modes'])
        jitted_get_fluctuating_term = jax.jit(self.get_fluctuating_term_maps)
        jitted_solve_wiener_filter_term = jax.jit(self.solve_generalized_wiener_filter_term)
        # jitted_get_conditional_proba_C_from_r = jax.jit(self.get_conditional_proba_C_from_r)
        jitted_single_Metropolis_Hasting_step_r = jax.jit(single_Metropolis_Hasting_step, static_argnames=['log_proba'])
        jitted_single_Metropolis_Hasting_step_B_f = jax.jit(single_Metropolis_Hasting_step, static_argnames=['log_proba','slow_ver','fullsky_ver','with_prints'])
        # jitted_get_conditional_proba_mixing_matrix_v2_JAX = jax.jit(self.get_conditional_proba_mixing_matrix_v2_JAX)
        # jitted_get_conditional_proba_mixing_matrix_v2_JAX = jax.jit(self.get_conditional_proba_mixing_matrix_v2_slow_JAX)
        # print("Using slow version of B_f sampling !", flush=True)
        jitted_get_conditional_proba_mixing_matrix_v2_JAX = jax.jit(self.get_conditional_proba_mixing_matrix_v2_slow_JAX_alt)
        print("Using ALT slow version of B_f sampling !", flush=True)
        if self.biased_version:
            print("Using biased version of mixing matrix sampling !!!", flush=True)
            jitted_get_conditional_proba_mixing_matrix_v2_JAX = jax.jit(self.get_biased_conditional_proba_mixing_matrix_v2_slow_JAX)
            # jitted_get_conditional_proba_mixing_matrix_v2_JAX = jax.jit(self.get_biased_conditional_proba_mixing_matrix_v2_slow_JAX_alt)
        
        jitted_get_conditional_proba_mixing_matrix_v1_JAX = jax.jit(self.get_conditional_proba_mixing_matrix_v1_slow_JAX_alt_harm)
        if self.harmonic_correction:
            print("Using harmonic correction for mixing matrix sampling !!!", flush=True)
            jitted_func_to_use = jax.jit(self.get_conditional_proba_mixing_matrix_v1_slow_JAX_alt_harm)
        else:
            jitted_func_to_use = jitted_get_conditional_proba_mixing_matrix_v2_JAX

        dimension_param_B_f = (self.number_frequencies-len_pos_special_freqs)*(self.number_correlations-1)


        _all_r_samples = jnp.zeros(actual_number_of_iterations+1)
        _all_B_f_samples = jnp.zeros((actual_number_of_iterations+1, dimension_param_B_f))
        number_correlations = self.number_correlations

        @scan_tqdm(actual_number_of_iterations)
        def all_sampling_steps(carry, iteration):
            
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample, PRNGKey = carry
            # eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, _all_r_samples, params_mixing_matrix_sample, PRNGKey = carry
            eta_maps_sample, WF_term_maps, fluct_maps, red_cov_matrix_sample, _all_r_samples, _all_B_f_samples, PRNGKey = carry

            PRNGKey, subPRNGKey = random.split(PRNGKey)
            
            # params_mixing_matrix_sample = _all_B_f_samples[iteration].reshape((self.number_frequencies-len_pos_special_freqs,number_correlations),order='F')
            params_mixing_matrix_sample = _all_B_f_samples[iteration].reshape((self.number_frequencies-len_pos_special_freqs,number_correlations-1),order='F')
            self.mixing_matrix_obj.update_params(params_mixing_matrix_sample)
            mixing_matrix_sampled = jnp.copy(self.mixing_matrix_obj.get_B(jax_use=True))
            chx.assert_axis_dimension(mixing_matrix_sampled, 0, self.number_frequencies)
            chx.assert_axis_dimension(mixing_matrix_sampled, 1, self.number_components)

            # Application of new mixing matrix
            BtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, mixing_matrix_sampled, jax_use=True)
            BtinvN_sqrt = get_BtinvN(jnp.sqrt(self.freq_inverse_noise), mixing_matrix_sampled, jax_use=True)
            s_cML = get_Wd(self.freq_inverse_noise, mixing_matrix_sampled, input_freq_maps, jax_use=True)[0]

            
            if self.sample_eta_B_f:
                map_random_x = jnp.empty(0)
                map_random_y = jnp.empty(0)
                time_start_sampling_eta_maps = time.time()
                eta_maps_sample = self.get_sampling_eta_v2(red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, subPRNGKey, map_random_x=map_random_x, map_random_y=map_random_y, suppress_low_modes=True)
                # eta_maps_sample = jitted_sample_eta(red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, subPRNGKey+1, map_random_x=map_random_x, map_random_y=map_random_y, suppress_low_modes=True)
                time_sampling_eta_maps = (time.time()-time_start_sampling_eta_maps)/60
                print("##### Sampling eta_maps at iteration {} in {} minutes".format(iteration+1, time_sampling_eta_maps), flush=True)

            if self.only_select_Bmodes:
                eta_maps_sample = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(jnp.copy(eta_maps_sample), red_cov_select_Bmodes, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

            # assert eta_maps_sample.shape[0] == self.nstokes
            # assert eta_maps_sample.shape[1] == self.npix
            chx.assert_shape(eta_maps_sample, (self.nstokes, self.npix))

            # Sampling step 2 : sampling of Gaussian variable s_c 
            initial_wiener_filter_term = jnp.copy(WF_term_maps)
            time_start_sampling_s_c_WF = time.time()
            wiener_filter_term = self.solve_generalized_wiener_filter_term(s_cML, red_cov_matrix_sample, BtinvNB, initial_guess=initial_wiener_filter_term)
            # wiener_filter_term = jitted_solve_wiener_filter_term(s_cML, red_cov_matrix_sample, BtinvNB, initial_guess=initial_wiener_filter_term)
            time_sampling_s_c_WF = (time.time()-time_start_sampling_s_c_WF)/60
            print("##### Sampling s_c_WF at iteration {} in {} minutes".format(iteration+1, time_sampling_s_c_WF), flush=True)

            # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(self.nside), size=(self.nstokes,self.npix))
            # map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(self.nside), size=(self.number_frequencies,self.nstokes,self.npix))
            
            subPRNGKey, new_subPRNGKey = random.split(subPRNGKey)

            map_random_realization_xi = jnp.empty(0)
            map_random_realization_chi = jnp.empty(0)
            time_start_sampling_s_c_fluct = time.time()
            fluctuation_maps = self.get_fluctuating_term_maps(red_cov_matrix_sample, BtinvNB, BtinvN_sqrt, new_subPRNGKey, map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, initial_guess=jnp.copy(fluct_maps))
            # fluctuation_new_maps = jitted_get_fluctuating_term(red_cov_matrix_sample, BtinvNB, BtinvN_sqrt, subPRNGKey+2, map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, initial_guess=jnp.copy(fluct_maps))
            time_sampling_s_c_fluct = (time.time()-time_start_sampling_s_c_fluct)/60
            print("##### Sampling s_c_fluct at iteration {} in {} minutes".format(iteration+1, time_sampling_s_c_fluct), flush=True)

            if self.only_select_Bmodes:
                fluctuation_maps = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(jnp.copy(fluctuation_maps), red_cov_select_Bmodes, nside=self.nside, lmin=self.lmin, n_iter=self.n_iter)

            s_c_sample = wiener_filter_term + fluctuation_maps

            chx.assert_shape(wiener_filter_term, (self.nstokes, self.npix))
            chx.assert_shape(fluctuation_maps, (self.nstokes, self.npix))
            chx.assert_shape(s_c_sample, (self.nstokes, self.npix))

            # Sampling step 3 : sampling of CMB covariance C
            
            ## Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
            c_ells_Wishart = get_cell_from_map_jax(s_c_sample, lmax=self.lmax, n_iter=self.n_iter)
            # c_ells_Wishart_modified = jnp.copy(c_ells_Wishart)
            # # for i in range(self.nstokes):
            # for i in range(c_ells_Wishart.shape[0]):
            #     # c_ells_Wishart_modified[i] *= 2*jnp.arange(self.lmax+1) + 1
            #     c_ells_Wishart_modified = c_ells_Wishart_modified.at[i].set(c_ells_Wishart_modified[i] * (2*jnp.arange(self.lmax+1) + 1))
            c_ells_Wishart_modified = jnp.copy(c_ells_Wishart)*(2*jnp.arange(self.lmax+1) + 1)
            red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell_jax(c_ells_Wishart_modified)[self.lmin:]


            time_start_sampling_C = time.time()
            # Sampling with Wishart
            if self.sample_C_inv_Wishart:
                red_cov_matrix_sample = self.get_inverse_wishart_sampling_from_c_ells(jnp.copy(c_ells_Wishart_modified), l_min=self.lmin)#[lmin:]

            elif self.sample_r_Metropolis:
                # r_all_samples = get_sample_parameter(mcmc_kernel_log_proba_r, r_sample, random_PRNGKey=subPRNGKey+iteration+3,
                #                                      red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                # chx.assert_shape(r_all_samples, (self.n_walkers_Metropolis, self.number_steps_sampler_r))
                # r_sample = r_all_samples[0,-1]
                # if iteration < num_sample_AM:
                #     step_size_r = self.step_size_r
                # else:
                #     step_size_r = jnp.sqrt(scale_param*(jnp.var(_all_r_samples[:iteration+1]) + epsilon_cov))

                # mean_r_samples = _all_r_samples.sum()/(iteration+1)
                # # variance_r_samples = ((_all_r_samples - mean_r_samples)**2).sum()/(iteration+1)
                # variance_r_samples = ((_all_r_samples - mean_r_samples)**2).sum()/iteration
                # adaptative_step_size = jnp.sqrt(self.scale_param*(variance_r_samples + self.epsilon_cov))
                # step_size_r = jnp.where(iteration<self.num_sample_AM,  self.step_size_r, adaptative_step_size)
                step_size_r = jnp.where(iteration<self.num_sample_AM,  self.step_size_r, jnp.sqrt(self.get_new_covariance_1d(iteration, _all_r_samples)))

                new_subPRNGKey, new_subPRNGKey_2 = random.split(new_subPRNGKey)

                # r_sample = single_Metropolis_Hasting_step(random_PRNGKey=subPRNGKey+3, old_sample=r_sample, 
                                                        #   step_size=step_size_r, log_proba=self.get_conditional_proba_C_from_r, 
                                                        #   red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                r_sample = single_Metropolis_Hasting_step(random_PRNGKey=new_subPRNGKey_2, old_sample=_all_r_samples[iteration],
                                                          step_size=step_size_r, log_proba=self.get_conditional_proba_C_from_r, 
                                                          red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                # r_sample = jitted_single_Metropolis_Hasting_step_r(random_PRNGKey=subPRNGKey+3, old_sample=r_sample, 
                #                                           step_size=self.step_size_r, log_proba=self.get_conditional_proba_C_from_r, 
                #                                           red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                # self.all_samples_r = self.update_variable(self.all_samples_r,r_sample)
                _all_r_samples = _all_r_samples.at[iteration+1].set(r_sample)

                red_cov_matrix_sample = theoretical_red_cov_r0_total + r_sample*theoretical_red_cov_r1_tensor
                print(f"## r sample : {r_sample}",flush=True)
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')
            time_sampling_C = (time.time()-time_start_sampling_C)/60

            print("##### Sampling C at iteration {} in {} minutes".format(iteration+1, time_sampling_C), flush=True)

            if red_cov_matrix_sample.shape[0] == self.lmax + 1:
                red_cov_matrix_sample = red_cov_matrix_sample[self.lmin:]
            chx.assert_shape(red_cov_matrix_sample, (self.lmax + 1 - self.lmin, self.nstokes, self.nstokes))
            
            # Preparation of sampling step 4
            extended_CMB_maps = jnp.zeros((self.number_components, self.nstokes, self.npix))
            extended_CMB_maps = extended_CMB_maps.at[0].set(s_c_sample)
            full_data_without_CMB = input_freq_maps - jnp.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
            chx.assert_shape(full_data_without_CMB, (self.number_frequencies, self.nstokes, self.npix))

            new_subPRNGKey_2, new_subPRNGKey_3 = random.split(new_subPRNGKey_2)
            # Sampling step 4
            if self.sample_eta_B_f:
                print("B_f sample :", params_mixing_matrix_sample, flush=True)
                # if iteration < num_sample_AM:
                #     # covariance_matrix_B_f = jnp.diag(initial_step_size_B_f.ravel(order='F'))
                #     covariance_matrix_B_f = jnp.diag(self.step_size_B_f.ravel(order='F')**2)
                # elif iteration == num_sample_AM:
                #     # step_size_B_f = jnp.sqrt(scale_param*(jnp.var(all_B_f_sample[:iteration],axis=0) + epsilon_cov))
                    
                #     # mean_sample_t_m1 = jnp.mean(all_B_f_sample[:iteration-1],axis=0).ravel(order='F')
                #     mean_sample_t = jnp.mean(_all_B_f_samples[:iteration],axis=0).ravel(order='F')
                    
                #     empirical_covariance = get_empirical_covariance_JAX(_all_B_f_samples[:iteration].reshape((iteration,(self.number_frequencies-len_pos_special_freqs)*2),order='F'))
                #     covariance_matrix_B_f = scale_param*(empirical_covariance + epsilon_cov*jnp.eye(dimension_param_B_f))
                # else:
                #     mean_sample_t_m1 = np.copy(mean_sample_t)
                #     mean_sample_t = (iteration/(iteration+1))*mean_sample_t_m1 + _all_B_f_samples[iteration].ravel(order='F')/(iteration+1)
                #     covariance_matrix_B_f = ((iteration-1)/iteration)*covariance_matrix_B_f + (scale_param/iteration)*(jnp.einsum('iteration,j->ij', _all_B_f_samples[iteration].ravel(order='F'), _all_B_f_samples[iteration].ravel(order='F'))
                #                             + iteration*jnp.einsum('iteration,j->ij',mean_sample_t_m1,mean_sample_t_m1)
                #                             - (iteration+1)*jnp.einsum('iteration,j->ij',mean_sample_t,mean_sample_t)
                #                             + epsilon_cov*jnp.eye(dimension_param_B_f))
                
                # empirical_covariance = jnp.where(iteration < num_sample_AM, 
                #                                  jnp.zeros((dimension_param_B_f,dimension_param_B_f)), 
                #                                  get_empirical_covariance_JAX(_all_B_f_samples[:iteration].reshape((iteration,(self.number_frequencies-len_pos_special_freqs)*2),order='F')))
                


                # mean_samples = _all_B_f_samples.sum(axis=0)/(iteration+1)

                # empirical_covariance = (jnp.einsum('ti,tj->tij',_all_B_f_samples,_all_B_f_samples).sum(axis=0) 
                #                         - (iteration+1)*jnp.einsum('i,j->ij',mean_samples,mean_samples))/(iteration)

                # covariance_matrix_B_f_AM = (self.scale_param/dimension_param_B_f)*(empirical_covariance + self.epsilon_cov*jnp.eye(dimension_param_B_f))
                
                # covariance_matrix_B_f = jnp.where(iteration < self.num_sample_AM, 
                #                                   jnp.diag(self.step_size_B_f.ravel(order='F')**2),
                #                                   self.get_new_covariance(iteration, _all_B_f_samples))
                covariance_matrix_B_f = jnp.where(iteration < self.num_sample_AM, 
                                                  self.covariance_step_size_B_f,
                                                  self.get_new_covariance(iteration, _all_B_f_samples))
                
                time_start_sampling_Bf = time.time()
                
                # params_mixing_matrix_sample = multivariate_Metropolis_Hasting_step(random_PRNGKey=new_subPRNGKey_3, old_sample=params_mixing_matrix_sample, 
                #                                           covariance_matrix=covariance_matrix_B_f, log_proba=jitted_get_conditional_proba_mixing_matrix_v1_JAX,
                #                                           full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, 
                #                                           red_cov_approx_matrix=red_cov_approx_matrix)
                params_mixing_matrix_sample = multivariate_Metropolis_Hasting_step(random_PRNGKey=new_subPRNGKey_3, old_sample=params_mixing_matrix_sample, 
                                                          covariance_matrix=covariance_matrix_B_f, log_proba=jitted_func_to_use,
                                                          full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, 
                                                          red_cov_approx_matrix=red_cov_approx_matrix)
                time_sampling_Bf = (time.time()-time_start_sampling_Bf)/60
                print("##### Sampling B_f at iteration {} in {} minutes".format(iteration+1, time_sampling_Bf), flush=True)

                _all_B_f_samples = _all_B_f_samples.at[iteration+1].set(params_mixing_matrix_sample.ravel(order='F'))
                chx.assert_axis_dimension(params_mixing_matrix_sample, 0, self.number_frequencies-len_pos_special_freqs)
                # chx.assert_axis_dimension(params_mixing_matrix_sample, 1, self.number_correlations-1)
                # assert len(params_mixing_matrix_sample.shape) == 2
                # Recording of the samples

            # Few tests to verify everything's fine
            # all_eigenvalues = np.linalg.eigh(red_cov_matrix_sample[self.lmin:])[0]
            # assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_cov_matrix_sample[self.lmin:])[0])>10**(-15)]>0)

            # # Recording of the samples
            # all_eta[iteration+1,...] = eta_maps_sample
            # # Recording of the samples
            # all_s_c_WF_maps[iteration+1,...] = wiener_filter_term
            # all_s_c_fluct_maps[iteration+1,...] = fluctuation_new_maps
            # # Recording of the samples
            # all_cell_samples[iteration+1,:,self.lmin:] = get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)
            # all_r_samples[iteration+1,...] = r_sample
            # all_params_mixing_matrix_samples[iteration+1,...] = params_mixing_matrix_sample

            # new_carry = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample, subPRNGKey)
            # all_samples = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample)

            # new_carry = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, _all_r_samples, params_mixing_matrix_sample, new_subPRNGKey_3)
            new_carry = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, _all_r_samples, _all_B_f_samples, new_subPRNGKey_3)
            all_samples = (eta_maps_sample, wiener_filter_term, fluctuation_maps, red_cov_matrix_sample, r_sample, params_mixing_matrix_sample)

            # if iteration%50 == 0:
            #     print("### Iteration n°", iteration, flush=True)
            print("### Iteration n°", iteration, flush=True)
            return new_carry, all_samples
                
        _all_r_samples = _all_r_samples.at[0].set(initial_guess_r)
        _all_B_f_samples = _all_B_f_samples.at[0].set(params_mixing_matrix_init_sample.ravel(order='F'))
        # initial_carry = (initial_eta, 
        #                  wiener_filter_term, fluctuation_maps, 
        #                  red_cov_matrix,
        #                  initial_guess_r,
        #                  params_mixing_matrix_init_sample,
        #                  PRNGKey)
        # initial_carry = (initial_eta, 
        #                  wiener_filter_term, fluctuation_maps, 
        #                  red_cov_matrix,
        #                  _all_r_samples,
        #                  params_mixing_matrix_init_sample,
        #                  PRNGKey)
        initial_carry = (initial_eta, 
                         wiener_filter_term, fluctuation_maps, 
                         red_cov_matrix,
                         _all_r_samples,
                         _all_B_f_samples,
                         PRNGKey)
        initial_carry_0 = (initial_eta, 
                         wiener_filter_term, fluctuation_maps, 
                         red_cov_matrix,
                         initial_guess_r,
                         params_mixing_matrix_init_sample,
                         PRNGKey)
        self.update_one_sample(initial_carry_0)

        time_start_sampling = time.time()
        # Start sampling !!!
        last_sample, all_samples = jlx.scan(all_sampling_steps, initial_carry, jnp.arange(actual_number_of_iterations))
        time_full_chain = (time.time()-time_start_sampling)/60      
        print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)


        # self.dict_last_samples = {'input_freq_maps':input_freq_maps, 'c_ell_approx':c_ell_approx, 
        #                      'CMB_c_ell':all_cell_samples[-1,...], 
        #                      'init_params_mixing_matrix':all_params_mixing_matrix_samples[-1,...],
        #                      'initial_guess_r':all_r_samples[-1], 
        #                      'initial_wiener_filter_term':all_s_c_WF_maps[-1,...], 
        #                      'initial_fluctuation_maps':all_s_c_fluct_maps[-1,...],
        #                      'theoretical_r0_total':theoretical_r0_total, 
        #                      'theoretical_r1_tensor':theoretical_r1_tensor}

        self.update_samples(all_samples)

        # all_eta, all_s_c_WF_maps, all_s_c_fluct_maps, all_cell_samples, all_r_samples, all_params_mixing_matrix_samples = all_samples

        # return all_eta, all_s_c_WF_maps, all_s_c_fluct_maps, all_cell_samples, all_r_samples, all_params_mixing_matrix_samples

    def restart_sampling(self, number_iterations_to_perform):
        """ 
            Perform number_iterations_to_perform new iterations
        """
        
        self.number_iterations_done = self.number_iterations_sampling
        self.number_iterations_sampling = self.number_iterations_done + number_iterations_to_perform

        print("Changing seed for new computations", flush=True)
        self.seed = self.seed + self.number_iterations_done
        
        return self.perform_sampling(self.dict_last_samples)


def create_MICMAC_sampler_from_toml_file(path_toml_file):
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()
    # dictionary_parameters = toml.load(path_toml_file)

    if dictionary_parameters['instrument_name'] != 'customized_instrument':
        instrument = fgbuster.get_instrument(dictionary_parameters['instrument_name'])
        dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
        dictionary_parameters['freq_inverse_noise'] = get_noise_covar(instrument['depth_p'], dictionary_parameters['nside'])

    del dictionary_parameters['instrument_name']
    return MICMAC_Sampler(**dictionary_parameters)
