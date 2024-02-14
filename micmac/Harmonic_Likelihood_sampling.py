import os, sys, time
from collections import namedtuple
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
import jaxopt as jopt
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

class Harmonic_MICMAC_Sampler(Sampling_functions):
    def __init__(self, nside, lmax, nstokes, 
                 frequency_array, freq_noise_c_ell, pos_special_freqs=[0,-1], 
                 number_components=3, lmin=2,
                 n_iter=8,
                 mask=None,

                 biased_version=False,
                 
                 r_true=0,
                 
                 step_size_r=1e-4,
                 covariance_B_f=None,
                 indexes_free_Bf=False,

                 instrument_name='SO_SAT',
                 number_iterations_sampling=100, number_iterations_done=0,
                 thinning=1, number_chains_MH=1,
                 seed=0,
                 disable_chex=True):
        """ Non parametric likelihood sampling object
        """

        super(Harmonic_MICMAC_Sampler,self).__init__(nside=nside,lmax=lmax,nstokes=nstokes,lmin=lmin,
                                            freq_inverse_noise=None, freq_noise_c_ell=freq_noise_c_ell,
                                            frequency_array=frequency_array,
                                            pos_special_freqs=pos_special_freqs,number_components=number_components,
                                            n_iter=n_iter, 
                                            mask=mask)

        # Quick test parameters
        self.instrument_name = instrument_name
        self.biased_version = bool(biased_version)
        self.disable_chex = disable_chex
        if indexes_free_Bf is False:
            indexes_free_Bf = jnp.arange((self.number_frequencies-len(pos_special_freqs))*(self.number_correlations-1))
        self.indexes_free_Bf = jnp.array(indexes_free_Bf)

        # CMB parameters
        self.r_true = float(r_true)
        # assert freq_noise_c_ell.shape == (self.number_frequencies,self.number_frequencies,self.lmax+1-self.lmin)
        self.freq_noise_c_ell = freq_noise_c_ell

        # Metropolis-Hastings parameters
        self.covariance_B_f = covariance_B_f
        self.step_size_r = step_size_r
        self.thinning = thinning
        self.number_chains_MH = number_chains_MH

        # Sampling parameters
        self.number_iterations_sampling = int(number_iterations_sampling) # Maximum number of iterations for the sampling
        self.number_iterations_done = int(number_iterations_done) # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed

        # Samples preparation
        self.all_params_mixing_matrix_samples = jnp.empty(0)
        self.all_samples_r = jnp.empty(0)
        # self.all_samples_CMB_c_ell = jnp.empty(0)
    

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

    def update_variable(self, all_samples, new_samples_to_add):
        if jnp.size(all_samples) == 0:
            return new_samples_to_add
        elif jnp.size(new_samples_to_add.shape) == 1:
            return jnp.hstack([all_samples,new_samples_to_add])
        else:
            return jnp.vstack([all_samples,new_samples_to_add])

    # def update_one_sample(self, one_sample):
    #     indice_s_c = 1
    #     # if self.sample_eta_B_f:
    #     self.all_samples_eta = self.update_variable(self.all_samples_eta, one_sample[0].reshape([1]+list(one_sample[0].shape)))
    #     self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, one_sample[5].reshape([1]+list(one_sample[5].shape)))
    #     # else:
    #     #     indice_s_c = -1

    #     self.all_samples_wiener_filter_maps = self.update_variable(self.all_samples_wiener_filter_maps, one_sample[indice_s_c].reshape([1]+list(one_sample[indice_s_c].shape)))
    #     self.all_samples_fluctuation_maps = self.update_variable(self.all_samples_fluctuation_maps, one_sample[indice_s_c+1].reshape([1]+list(one_sample[indice_s_c+1].shape)))
    #     # self.all_samples_s_c = self.update_variable(self.all_samples_s_c, all_samples[indice_s_c]+all_samples[indice_s_c+1])

    #     if one_sample[indice_s_c+2].shape[0] == self.lmax+1-self.lmin:
    #         one_sample_CMB_c_ell = get_c_ells_from_red_covariance_matrix(one_sample[indice_s_c+2])
    #     else:
    #         one_sample_CMB_c_ell = one_sample[indice_s_c+2]
    #     self.all_samples_CMB_c_ell = self.update_variable(self.all_samples_CMB_c_ell, one_sample_CMB_c_ell.reshape([1]+list(one_sample_CMB_c_ell.shape)))
    #     self.all_samples_r = self.update_variable(self.all_samples_r, one_sample[indice_s_c+3])
        
    def update_samples_MH(self, all_samples):
        self.all_samples_r = self.update_variable(self.all_samples_r, all_samples[...,-1])

        self.all_params_mixing_matrix_samples = self.update_variable(self.all_params_mixing_matrix_samples, all_samples[...,:-1].reshape((all_samples.shape[0],self.number_frequencies-len(self.pos_special_freqs),self.number_components-1),order='F'))

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


    def get_alm_from_frequency_maps(self, input_freq_maps):

        assert len(input_freq_maps.shape) == 3
        assert input_freq_maps.shape[0] == self.number_frequencies
        assert input_freq_maps.shape[1] == self.nstokes
        assert input_freq_maps.shape[2] == self.npix

        ## Preparing input alms data
        def wrapper_map2alm(maps_, lmax=self.lmax, n_iter=self.n_iter, nside=self.nside):
            alm_T, alm_E, alm_B = hp.map2alm(
                maps_.reshape((3, 12 * nside**2)), lmax=lmax, iter=n_iter
            )
            return np.array([alm_T, alm_E, alm_B])

        @partial(jax.jit, static_argnums=(1))
        def pure_call_map2alm(maps_, lmax):
            shape_output = (3, (lmax + 1) * (lmax // 2 + 1))
            return jax.pure_callback(
                wrapper_map2alm,
                jax.ShapeDtypeStruct(shape_output, np.complex128),
                maps_.ravel(),
            )

        JAX_input_freq_maps = jnp.array(input_freq_maps)
        def get_freq_alm(num_frequency):
            input_map_extended = jnp.vstack((JAX_input_freq_maps[num_frequency,0], JAX_input_freq_maps[num_frequency,...]))
            all_alms = jnp.array(pure_call_map2alm(input_map_extended, lmax=self.lmax))
            return all_alms[3-self.nstokes:,...]
  
        return jax.vmap(get_freq_alm)(jnp.arange(self.number_frequencies))


    def perform_harmonic_minimize(self, input_freq_maps, c_ell_approx, init_params_mixing_matrix,
                                    theoretical_r0_total, theoretical_r1_tensor,
                                    initial_guess_r=0,
                                    method_used='ScipyMinimize',
                                    **options_minimizer):
        """ Perform a minimization to find the best r and B_f in harmonic domain, given the other parameters
        """


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

        ## Testing the initial spectra given in case the sampling is done with r
        
        assert len(theoretical_r0_total.shape) == 2
        assert (theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin) #or (theoretical_r0_total.shape[1] == self.lmax + 1)
        assert len(theoretical_r1_tensor.shape) == 2
        assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

        ## Testing the initial CMB spectra given
        # if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):    
        #     CMB_c_ell = CMB_c_ell[indices_to_consider,:]
        #     c_ell_approx = c_ell_approx[indices_to_consider,:]
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[indices_to_consider,:]

        ## Testing the initial mixing matrix
        if len(init_params_mixing_matrix.shape) == 1:
            assert len(init_params_mixing_matrix) == (self.number_frequencies-len(self.pos_special_freqs))*(self.number_correlations-1)
        else:
            # assert len(init_params_mixing_matrix.shape) == 2
            assert init_params_mixing_matrix.shape[0] == (self.number_frequencies-len(self.pos_special_freqs))
            assert init_params_mixing_matrix.shape[1] == (self.number_correlations-1)

        red_cov_approx_matrix = get_reduced_matrix_from_c_ell_jax(c_ell_approx)[self.lmin:,...]
        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
        theoretical_red_cov_r1_tensor  = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)

        ## Getting alms from the input maps
        freq_alms_input_maps = self.get_alm_from_frequency_maps(input_freq_maps)
        freq_red_inverse_noise = jnp.einsum('fgl,sk->fglsk', self.freq_noise_c_ell, jnp.eye(self.nstokes))
        noise_weighted_alm_data = frequency_alms_x_obj_red_covariance_cell_JAX(freq_alms_input_maps, freq_red_inverse_noise, lmin=self.lmin)

        
        # Setting up the JAXOpt class:
        if method_used in ['BFGS', 'GradientDescent', 'LBFGS', 'NonlinearCG', 'ScipyMinimize']:
            class_solver = getattr(jopt, method_used)
        else:
            raise ValueError("Method used not recognized for minimization")

        func_to_minimize = lambda sample_B_f_r: -self.harmonic_marginal_probability(sample_B_f_r,
                                                            noise_weighted_alm_data=noise_weighted_alm_data, 
                                                            red_cov_approx_matrix=red_cov_approx_matrix, 
                                                            theoretical_red_cov_r0_total=theoretical_red_cov_r0_total, 
                                                            theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor)
        optimizer = class_solver(fun=func_to_minimize, **options_minimizer)

        # Preparing the initial parameters
        init_params_B_f_r = jnp.concatenate((init_params_mixing_matrix.ravel(order='F'), 
                                            jnp.array(initial_guess_r).reshape(1)))
        
        print("Start of minimization", flush=True)
        params, state = optimizer.run(init_params_B_f_r,
                                        )
        print("End of minimization", flush=True)
        print("Found parameters", params, flush=True)
        print("With state", state, flush=True)
        return params


    def perform_harmonic_MH(self, input_freq_maps, c_ell_approx, init_params_mixing_matrix, theoretical_r0_total, theoretical_r1_tensor,
                            initial_guess_r=0, covariance_B_f_r=None):
        """ Perform Metropolis Hastings to find the best r and B_f in harmonic domain, given the other parameters

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
            indices_to_consider = np.arange(6) # All auto- and cross-correlations

        ## Testing the initial spectra given in case the sampling is done with r
        
        assert len(theoretical_r0_total.shape) == 2
        assert (theoretical_r0_total.shape[1] == self.lmax + 1 - self.lmin) #or (theoretical_r0_total.shape[1] == self.lmax + 1)
        assert len(theoretical_r1_tensor.shape) == 2
        assert theoretical_r1_tensor.shape[1] == theoretical_r0_total.shape[1]

        theoretical_red_cov_r0_total = get_reduced_matrix_from_c_ell(theoretical_r0_total)
        theoretical_red_cov_r1_tensor  = get_reduced_matrix_from_c_ell(theoretical_r1_tensor)

        ## Testing the initial CMB spectra given
        # if self.nstokes == 2 and (CMB_c_ell.shape[0] != len(indices_to_consider)):    
        #     CMB_c_ell = CMB_c_ell[indices_to_consider,:]
        #     c_ell_approx = c_ell_approx[indices_to_consider,:]
        if self.nstokes == 2 and (c_ell_approx.shape[0] != len(indices_to_consider)):
            c_ell_approx = c_ell_approx[indices_to_consider,:]
        
        ## Testing the initial mixing matrix
        try:
            assert init_params_mixing_matrix.shape[0] == self.number_chains_MH
        except:
            if len(init_params_mixing_matrix) == 1:
                init_params_mixing_matrix = np.repeat(init_params_mixing_matrix, 
                                                    self.number_chains_MH).reshape(
                                                        (self.number_chains_MH, 
                                                        init_params_mixing_matrix.shape[0]), order='F')
            else:
                init_params_mixing_matrix = np.repeat(init_params_mixing_matrix, 
                                                    self.number_chains_MH).reshape(
                                                        (self.number_chains_MH, 
                                                        init_params_mixing_matrix.shape[0],
                                                        init_params_mixing_matrix.shape[1]), order='F')

        if self.number_chains_MH == 1:
            if len(init_params_mixing_matrix.shape) == 1:
                assert len(init_params_mixing_matrix) == (self.number_frequencies-len(self.pos_special_freqs))*(self.number_correlations-1)
            else:
                # assert len(init_params_mixing_matrix.shape) == 2
                assert init_params_mixing_matrix.shape[0] == (self.number_frequencies-len(self.pos_special_freqs))
                assert init_params_mixing_matrix.shape[1] == (self.number_correlations-1)
        
        else:
            if len(init_params_mixing_matrix.shape) == 2:
                assert init_params_mixing_matrix.shape[1] == (self.number_frequencies-len(self.pos_special_freqs))*(self.number_correlations-1)
            else:
                assert init_params_mixing_matrix.shape[1] == (self.number_frequencies-len(self.pos_special_freqs))
                assert init_params_mixing_matrix.shape[2] == (self.number_correlations-1)
        try: 
            assert jnp.size(initial_guess_r) == self.number_chains_MH
        except:
            initial_guess_r = np.repeat(initial_guess_r, self.number_chains_MH).reshape(self.number_chains_MH)

        ## Final set of tests
        # assert len(CMB_c_ell.shape) == 2
        # assert CMB_c_ell.shape[1] == self.lmax + 1
        assert len(c_ell_approx.shape) == 2
        assert c_ell_approx.shape[1] == self.lmax + 1

        # Preparing for the full Gibbs sampling
        len_pos_special_freqs = len(self.pos_special_freqs)

        ## Initial guesses
        initial_eta = jnp.zeros((self.nstokes,self.npix))
        params_mixing_matrix_init_sample = jnp.copy(init_params_mixing_matrix).reshape(
                                            ((self.number_frequencies-len_pos_special_freqs),self.number_correlations-1), order='F')

        ## CMB covariance preparation in the format [lmax,nstokes,nstokes]
        red_cov_approx_matrix = get_reduced_matrix_from_c_ell_jax(c_ell_approx)[self.lmin:,...]
        
        ## Preparation of the mixing matrix
        self.mixing_matrix_obj = MixingMatrix(self.frequency_array, self.number_components, params_mixing_matrix_init_sample, pos_special_freqs=self.pos_special_freqs)

        ## Preparing the scalar quantities
        PRNGKey = random.PRNGKey(self.seed)

        dimension_param_B_f = (self.number_frequencies-len_pos_special_freqs)*(self.number_correlations-1)
        number_correlations = self.number_correlations

        ## Preparing the step-size for Metropolis-within-Gibbs of B_f sampling
        if covariance_B_f_r is None:
            if self.covariance_B_f is None:
                raise ValueError("Please provide a covariance_B_f")
                assert (self.covariance_B_f).shape == ((self.number_frequencies-len(pos_special_freqs))*(self.number_correlations-1),(self.number_frequencies-len(pos_special_freqs))*(self.number_correlations-1))
        
            covariance_B_f_r = jnp.zeros((dimension_param_B_f+1, dimension_param_B_f+1))
            covariance_B_f_r = covariance_B_f_r.at[:dimension_param_B_f,:dimension_param_B_f].set(self.covariance_B_f)
            covariance_B_f_r = covariance_B_f_r.at[dimension_param_B_f,dimension_param_B_f].set(self.step_size_r**2)
        else:
            assert covariance_B_f_r.shape == (dimension_param_B_f+1,dimension_param_B_f+1)

        print('Covariance B_f, r', covariance_B_f_r, flush=True)

        
        input_freq_alms = self.get_alm_from_frequency_maps(input_freq_maps)
        freq_red_inverse_noise = jnp.einsum('fgl,sk->fglsk', self.freq_noise_c_ell, jnp.eye(self.nstokes))
        noise_weighted_alm_data = frequency_alms_x_obj_red_covariance_cell_JAX(input_freq_alms, freq_red_inverse_noise, lmin=self.lmin)

        print(f"Starting {self.number_iterations_sampling} iterations for harmonic run", flush=True)

        MHState = namedtuple("MHState", ["u", "rng_key"])

        class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
            sample_field = "u"

            def __init__(self, log_proba, covariance_matrix):
                self.log_proba = log_proba
                self.covariance_matrix = covariance_matrix

            def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
                return MHState(init_params, rng_key)

            def sample(self, state, model_args, model_kwargs):
                
                new_sample, rng_key = multivariate_Metropolis_Hasting_step_numpyro(state, 
                                covariance_matrix=self.covariance_matrix, 
                                log_proba=self.log_proba, 
                                **model_kwargs)
                return MHState(new_sample, rng_key)

        mcmc_obj = MCMC(MetropolisHastings(log_proba=self.harmonic_marginal_probability, covariance_matrix=covariance_B_f_r), 
                        num_warmup=0, 
                        num_samples=self.number_iterations_sampling, 
                        thinning=self.thinning, 
                        num_chains=self.number_chains_MH, 
                        progress_bar=True)

        # Initializing r and B_f samples
        init_params_mixing_matrix_r = jnp.concatenate((params_mixing_matrix_init_sample.ravel(order='F'), 
                                                        jnp.array(initial_guess_r).reshape(1)))

        time_start_sampling = time.time()
        mcmc_obj.run(PRNGKey, 
                        init_params=init_params_mixing_matrix_r, 
                        noise_weighted_alm_data=noise_weighted_alm_data, 
                        theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, 
                        theoretical_red_cov_r0_total=theoretical_red_cov_r0_total, 
                        red_cov_approx_matrix=red_cov_approx_matrix)

        time_full_chain = (time.time()-time_start_sampling)/60      
        print(f"End of MH iterations for harmonic run in {time_full_chain} minutes, now saving results !", flush=True)
        
        posterior_samples = mcmc_obj.get_samples()
        print("Test", posterior_samples)
        print("Test size", posterior_samples.shape, flush=True)
        mcmc_obj.print_summary()

        # Saving the samples as attributes of the Sampler object
        self.update_samples_MH(posterior_samples)
        self.number_iterations_done = self.number_iterations_sampling





def create_Harmonic_MICMAC_sampler_from_toml_file(path_toml_file):
    """ Create a Harmonic_MICMAC_Sampler object from the path of a toml file 
    """
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    if dictionary_parameters['instrument_name'] != 'customized_instrument':
        instrument = fgbuster.get_instrument(dictionary_parameters['instrument_name'])
        dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
        # dictionary_parameters['freq_inverse_noise'] = get_noise_covar(instrument['depth_p'], dictionary_parameters['nside'])
        dictionary_parameters['freq_noise_c_ell'] = get_true_Cl_noise(jnp.array(instrument['depth_p']), dictionary_parameters['lmax'])[...,dictionary_parameters['lmin']:]

    # del dictionary_parameters['instrument_name']
    return Harmonic_MICMAC_Sampler(**dictionary_parameters)
