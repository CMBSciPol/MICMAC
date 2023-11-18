import os, sys, time
import numpy as np
import healpy as hp
import scipy
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
import numpyro

from .tools import *
from .jax_tools import *
from .algorithm_toolbox import *
from .proba_functions import *
from .Sampling_toolbox import *
from .mixingmatrix import *
from .noisecovar import *
from .jax_Sampling_toolbox import *
from .temporary_tools import *

from jax import config
config.update("jax_enable_x64", True)

class MICMAC_Sampler(object):
    def __init__(self, nside, lmax, nstokes, frequency_array, freq_inverse_noise, 
                 number_components=3, lmin=2,
                 r_true=0, pos_special_freqs=[0,-1], only_select_Bmodes=False, no_Emodes_CMB=False, 
                 sample_eta_B_f=True, 
                 sample_r_Metropolis=True, sample_C_inv_Wishart=False,
                 n_iter=8, limit_iter_cg=2000, tolerance_CG=10**(-12),
                 n_walkers=1, step_size_B_f=10**(-4), step_size_r=10**(-4),
                 fullsky_ver=True, slow_ver=False,
                 number_steps_sampler_B_f=100, number_steps_sampler_r=100,
                 number_iterations_sampling=30, number_iterations_done=0, seed=0):
        """ Non parametric likelihood sampling object
        """

        # Quick test parameters
        self.fullsky_ver = bool(fullsky_ver)
        self.slow_ver = bool(slow_ver)
        self.sample_eta_B_f = bool(sample_eta_B_f)

        # CMB parameters
        self.freq_inverse_noise = freq_inverse_noise
        self.r_true = float(r_true)
        self.only_select_Bmodes = bool(only_select_Bmodes)
        self.no_Emodes_CMB = bool(no_Emodes_CMB)
        # assert np.abs(overrelaxation_param) <= 1
        # self.overrelaxation_param = float(overrelaxation_param)
        assert (sample_r_Metropolis or sample_C_inv_Wishart == True) and (sample_r_Metropolis and sample_C_inv_Wishart == False)
        self.sample_r_Metropolis = bool(sample_r_Metropolis)
        self.sample_C_inv_Wishart = bool(sample_C_inv_Wishart)

        # Problem parameters
        assert (2**np.log2(nside) == nside) or (nside == 1)
        self.nside = int(nside)
        self.lmax = int(lmax)
        assert nstokes in [1,2,3]
        self.nstokes = int(nstokes)
        self.lmin = int(lmin)
        self.n_iter = int(n_iter) # Number of iterations for Python estimation of alms
        self.frequency_array = frequency_array
        self.number_components = int(number_components)
        self.pos_special_freqs = pos_special_freqs

        # CG parameters
        self.limit_iter_cg = int(limit_iter_cg) # Maximum number of iterations for the different CGs
        self.tolerance_CG = float(tolerance_CG) # Tolerance for the different CGs

        # Metropolis-Hastings parameters
        self.n_walkers = int(n_walkers) # Number of walkers for the MCMC to sample the mixing matrix or r
        self.step_size_B_f = step_size_B_f
        self.step_size_r = step_size_r
        self.number_steps_sampler_B_f = int(number_steps_sampler_B_f) # Maximum number of steps for the Metropolis-Hasting to sample the mixing matrix
        self.number_steps_sampler_r = int(number_steps_sampler_r)

        # Sampling parameters
        self.number_iterations_sampling = int(number_iterations_sampling) # Maximum number of iterations for the sampling
        self.number_iterations_done = int(number_iterations_done) # Number of iterations already accomplished, in case the chain is resuming from a previous run
        self.seed = seed

    @property
    def npix(self):
        return 12*self.nside**2

    @property
    def number_correlations(self):
        """ Maximum number of correlations depending of the number of Stokes parameters : 
            6 (TT,EE,BB,TE,EB,TB) for 3 Stokes parameters ; 3 (EE,BB,EB) for 2 Stokes parameters ; 1 (TT) for 1 Stokes parameter"""
        return int(np.ceil(self.nstokes**2/2) + np.floor(self.nstokes/2))

    @property
    def number_frequencies(self):
        return len(self.frequency_array)

    @property
    def param_dict(self):
        return {'nside':self.nside, 'lmax':self.lmax, 'nstokes':self.nstokes, 'number_correlations':self.number_correlations,'number_frequencies':self.number_frequencies, 'number_components':self.number_components}


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

        return input_freq_maps, theoretical_red_cov_r0_total, theoretical_red_cov_r1_tensor

    # def sample_covariance(self, pixel_maps):
    #     """ Power spectrum sampling, given the sampled maps, following inverse Wishart distribution """
    #     c_ells_Wishart = get_cell_from_map(pixel_maps, lmax=self.lmax, n_iter=self.n_iter)
    #     return get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart, l_min=self.lmin, option_ell_2=self.option_ell_2)#[self.lmin:,...]


    def perform_sampling(self, input_freq_maps, c_ell_approx, CMB_covariance, init_params_mixing_matrix, 
                         initial_guess_r=10**(-8), initial_wiener_filter_term=np.empty(0), initial_fluctuation_maps=np.empty(0),
                         theoretical_r0_total=np.empty(0), theoretical_r1_tensor=np.empty(0)):
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

        len_pos_special_freqs = len(self.pos_special_freqs)

        actual_number_of_iterations = self.number_iterations_sampling + 1 - self.number_iterations_done
        all_eta = np.zeros((actual_number_of_iterations, self.nstokes, self.npix))
        all_s_c_WF_maps = np.zeros((actual_number_of_iterations, self.nstokes, self.npix))
        all_s_c_fluct_maps = np.zeros((actual_number_of_iterations, self.nstokes, self.npix))
        all_cell_samples = np.zeros((actual_number_of_iterations, self.number_correlations, self.lmax+1))
        all_r_samples = np.zeros(actual_number_of_iterations)
        all_params_mixing_matrix_samples = np.zeros((actual_number_of_iterations, self.number_frequencies-len_pos_special_freqs, self.number_correlations-1))

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
            wiener_filter_term = np.zeros((self.nstokes,self.npix))
        else:
            assert len(initial_wiener_filter_term.shape) == 2
            assert initial_wiener_filter_term.shape[0] == self.nstokes
            assert initial_wiener_filter_term.shape[1] == self.npix
            wiener_filter_term = initial_wiener_filter_term
        
        if len(initial_fluctuation_maps) == 0:
            fluctuation_maps = np.zeros((self.nstokes,self.npix))
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

       

        if self.nstokes == 2 and (CMB_covariance.shape[0] != len(indices_to_consider)):    
            CMB_covariance = CMB_covariance[indices_to_consider,:]
            c_ell_approx = c_ell_approx[indices_to_consider,:]
        
        assert len(CMB_covariance.shape) == 2
        assert CMB_covariance.shape[1] == self.lmax + 1
        assert len(c_ell_approx.shape) == 2
        assert c_ell_approx.shape[1] == self.lmax + 1

        # Initial guesses
        all_s_c_WF_maps[0,...] = wiener_filter_term
        all_s_c_fluct_maps[0,...] = fluctuation_maps
        all_cell_samples[0,...] = np.copy(CMB_covariance)
        params_mixing_matrix_sample = np.copy(init_params_mixing_matrix).reshape(((self.number_frequencies-len_pos_special_freqs),self.number_correlations-1), order='F')
        all_params_mixing_matrix_samples[0,...] = params_mixing_matrix_sample
        all_r_samples[0,...] = initial_guess_r

        
        # CMB covariance preparation
        red_cov_approx_matrix = get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin:,...]
        red_cov_matrix_sample = get_reduced_matrix_from_c_ell(CMB_covariance)[self.lmin:,...]

        mixing_matrix_obj = MixingMatrix(self.frequency_array, self.number_components, params_mixing_matrix_sample, pos_special_freqs=self.pos_special_freqs)
        mixing_matrix_sampled = mixing_matrix_obj.get_B()


        num_warmup = 0

        kernel_log_proba_Bf = MetropolisHastings_log(new_get_conditional_proba_full_likelihood_JAX_from_params, step_size=np.ravel(self.step_size_B_f,order='F'))
        mcmc_kernel_log_proba_Bf = numpyro.infer.MCMC(kernel_log_proba_Bf, num_chains=self.n_walkers, num_warmup=num_warmup, num_samples=self.number_steps_sampler_B_f)

        kernel_log_proba_r = MetropolisHastings_log(get_conditional_proba_C_from_r, step_size=self.step_size_r)
        mcmc_kernel_log_proba_r = numpyro.infer.MCMC(kernel_log_proba_r, num_chains=self.n_walkers, num_warmup=num_warmup, num_samples=self.number_steps_sampler_r)

        if not(self.sample_eta_B_f):
            print("Not sampling for eta and B_f, only for s_c and the covariance !", flush=True)
        if self.sample_r_Metropolis:
            print("Sample for r instead of C !", flush=True)
        else:
            print("Sample for C with inverse Wishart !", flush=True)
        print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done", flush=True)
        # print(f"Starting {self.number_iterations_sampling} iterations from {self.number_iterations_done} iterations done with {self.overrelaxation_param} overrelaxation parameter", flush=True)
        
        PRNGKey = random.PRNGKey(self.seed)
        
        time_start_sampling = time.time()
        # Start sampling !!!
        for iteration in range(self.number_iterations_sampling):
            print("### Start Iteration n°", iteration, flush=True)

            PRNGKey, subPRNGKey = random.split(PRNGKey)
            
            # Application of new mixing matrix
            BtinvNB = get_inv_BtinvNB(self.freq_inverse_noise, mixing_matrix_sampled)
            BtinvN_sqrt = get_BtinvN(np.sqrt(self.freq_inverse_noise), mixing_matrix_sampled)
            # s_cML = np.einsum('cd,df,fsp->s', BtinvNB, get_BtinvN(freq_inverse_noise, mixing_matrix_sampled), input_freq_maps)[0] # Computation of E^t (B^t N^{-1} B)^{-1} B^t N^{-1} d
            s_cML = get_Wd(self.freq_inverse_noise, mixing_matrix_sampled, input_freq_maps, jax_use=False)[0]
            # if only_select_Bmodes:
            #     s_cML = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(s_cML), red_cov_select_Bmodes, lmin=2, n_iter=n_iter)#[1:,...]
            
            # Sampling step 1 : Sample eta term with formulation :
            if self.sample_eta_B_f:
                map_random_x = []
                map_random_y = []
                time_start_sampling_eta_maps = time.time()
                eta_maps_sample = get_sampling_eta_v2(self.param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, lmin=self.lmin, n_iter=self.n_iter)

                if self.only_select_Bmodes:
                    eta_maps_sample = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(eta_maps_sample), red_cov_select_Bmodes, lmin=self.lmin, n_iter=self.n_iter)

                time_sampling_eta_maps = (time.time()-time_start_sampling_eta_maps)/60
                print("##### Sampling eta_maps at iteration {} in {} minutes".format(iteration+1, time_sampling_eta_maps), flush=True)

                # Recording of the samples
                all_eta[iteration+1,...] = eta_maps_sample

                assert eta_maps_sample.shape[0] == self.nstokes
                assert eta_maps_sample.shape[1] == self.npix


            # Sampling step 2 : sampling of Gaussian variable s_c 
            initial_wiener_filter_term = np.copy(wiener_filter_term)
            time_start_sampling_s_c_WF = time.time()
            wiener_filter_term = solve_generalized_wiener_filter_term(self.param_dict, s_cML, red_cov_matrix_sample, BtinvNB, initial_guess=initial_wiener_filter_term, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_CG)
            time_sampling_s_c_WF = (time.time()-time_start_sampling_s_c_WF)/60
            print("##### Sampling s_c_WF at iteration {} in {} minutes".format(iteration+1, time_sampling_s_c_WF), flush=True)

            initial_fluctuation_maps = np.copy(fluctuation_maps)
            # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(self.nside), size=(self.nstokes,self.npix))
            # map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(self.nside), size=(self.number_frequencies,self.nstokes,self.npix))
            map_random_realization_xi = np.empty(0)
            map_random_realization_chi = np.empty(0)
            time_start_sampling_s_c_fluct = time.time()
            fluctuation_maps = get_fluctuating_term_maps(self.param_dict, red_cov_matrix_sample, BtinvNB, BtinvN_sqrt, map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, initial_guess=initial_fluctuation_maps, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_CG)
            time_sampling_s_c_fluct = (time.time()-time_start_sampling_s_c_fluct)/60
            print("##### Sampling s_c_fluct at iteration {} in {} minutes".format(iteration+1, time_sampling_s_c_fluct), flush=True)

            if self.only_select_Bmodes:
                fluctuation_maps = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(fluctuation_maps), red_cov_select_Bmodes, lmin=self.lmin, n_iter=self.n_iter)#[1:,...]

            # Recording of the samples
            all_s_c_WF_maps[iteration+1,...] = wiener_filter_term
            all_s_c_fluct_maps[iteration+1,...] = fluctuation_maps

            s_c_sample = wiener_filter_term + fluctuation_maps

            # s_c_sample = wiener_filter_term + np.sqrt((1-self.overrelaxation_param**2))*fluctuation_maps + self.overrelaxation_param*(s_c_sample - wiener_filter_term)
            # WRONG : Actually need (N_c^-1 + C^-1)^{-1/2}, which is not trivial at all

            assert len(wiener_filter_term.shape) == 2
            assert wiener_filter_term.shape[0] == self.nstokes
            assert wiener_filter_term.shape[1] == self.npix
            assert len(fluctuation_maps.shape) == 2
            assert fluctuation_maps.shape[0] == self.nstokes
            assert fluctuation_maps.shape[1] == self.npix
            assert s_c_sample.shape[0] == self.nstokes
            assert s_c_sample.shape[1] == self.npix

            # Sampling step 3 : sampling of CMB covariance C
            
            # Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
            c_ells_Wishart = get_cell_from_map(s_c_sample, lmax=self.lmax, n_iter=self.n_iter)
            # if only_select_Bmodes:
            #     c_ells_Wishart[0,:] = 0
            #     c_ells_Wishart[2,:] = 0

            c_ells_Wishart_modified = np.copy(c_ells_Wishart)
            for i in range(self.nstokes):
                    c_ells_Wishart_modified[i] *= 2*np.arange(self.lmax+1) + 1
            red_c_ells_Wishart_modified = get_reduced_matrix_from_c_ell(c_ells_Wishart_modified)[self.lmin:]


            time_start_sampling_C = time.time()
            # Sampling with Wishart
            if self.sample_C_inv_Wishart:
                red_cov_matrix_sample = get_inverse_wishart_sampling_from_c_ells(np.copy(c_ells_Wishart), l_min=self.lmin)#[lmin:]

            elif self.sample_r_Metropolis:
                r_all_samples = get_sample_parameter(mcmc_kernel_log_proba_r, all_r_samples[iteration], random_PRNGKey=subPRNGKey+iteration+3, lmin=self.lmin, lmax=self.lmax, 
                                                     red_sigma_ell=red_c_ells_Wishart_modified, theoretical_red_cov_r1_tensor=theoretical_red_cov_r1_tensor, theoretical_red_cov_r0_total=theoretical_red_cov_r0_total)
                assert len(r_all_samples.shape) == 2
                assert r_all_samples.shape[0] == self.n_walkers
                assert r_all_samples.shape[1] == self.number_steps_sampler_r

                r_sample = r_all_samples[0,-1]
                red_cov_matrix_sample = theoretical_red_cov_r0_total + r_sample*theoretical_red_cov_r1_tensor
                print(f"## r sample : {r_sample}",flush=True)
            else:
                raise Exception('C not sampled in any way !!! It must be either inv Wishart or through r sampling !')
            time_sampling_C = (time.time()-time_start_sampling_C)/60

            print("##### Sampling C at iteration {} in {} minutes".format(iteration+1, time_sampling_C), flush=True)

            if red_cov_matrix_sample.shape[0] == self.lmax + 1:
                red_cov_matrix_sample = red_cov_matrix_sample[self.lmin:]
            assert red_cov_matrix_sample.shape[0] == self.lmax + 1 - self.lmin
            assert red_cov_matrix_sample.shape[1] == self.nstokes
            assert red_cov_matrix_sample.shape[2] == self.nstokes
            
            # Recording of the samples
            # all_cell_samples[iteration+1,...] = get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)
            all_cell_samples[iteration+1,:,self.lmin:] = get_c_ells_from_red_covariance_matrix(red_cov_matrix_sample)
            all_r_samples[iteration+1,...] = r_sample
            
            # Preparation of sampling step 4
            extended_CMB_maps = np.zeros((self.number_components, self.nstokes, self.npix))
            extended_CMB_maps[0] = s_c_sample
            full_data_without_CMB = input_freq_maps - np.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
            assert full_data_without_CMB.shape[0] == self.number_frequencies
            assert full_data_without_CMB.shape[1] == self.nstokes
            assert full_data_without_CMB.shape[2] == self.npix


            # Sampling step 4
            if self.sample_eta_B_f:
                time_start_sampling_Bf = time.time()
                # number_steps_sampler_random[iteration] = number_steps_sampler + np.random.randint(0,number_steps_sampler)
                # few_params_mixing_matrix_samples = get_sample_B_f(new_get_conditional_proba_full_likelihood_JAX_from_params, step_size_array.ravel(order='F'), number_steps_sampler_random[iteration], first_guess_params_mixing_matrix.ravel(), random_PRNGKey=jax.random.PRNGKey(100+iteration), n_walkers=n_walkers, num_warmup=num_warmup, pos_special_freqs=mixing_matrix_obj.pos_special_freqs, fullsky_ver=fullsky_ver, slow_ver=slow_ver, param_dict=param_dict, full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, freq_inverse_noise=freq_inverse_noise, red_cov_approx_matrix=red_cov_approx_matrix, lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance_CG, with_prints=False)
                print("B_f sample :", params_mixing_matrix_sample, flush=True)
                all_numpyro_mixing_matrix_samples = get_sample_parameter(mcmc_kernel_log_proba_Bf, params_mixing_matrix_sample.ravel(order='F'), random_PRNGKey=subPRNGKey+iteration+4, pos_special_freqs=self.pos_special_freqs, fullsky_ver=self.fullsky_ver, slow_ver=self.slow_ver, param_dict=self.param_dict, full_data_without_CMB=full_data_without_CMB, modified_sample_eta_maps=eta_maps_sample, freq_inverse_noise=self.freq_inverse_noise, red_cov_approx_matrix=red_cov_approx_matrix, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_CG, with_prints=False)
                params_mixing_matrix_sample = all_numpyro_mixing_matrix_samples[0,-1,:].reshape((self.number_frequencies-len_pos_special_freqs,2), order='F')
                time_sampling_Bf = (time.time()-time_start_sampling_Bf)/60
                print("##### Sampling B_f at iteration {} in {} minutes".format(iteration+1, time_sampling_Bf), flush=True)

                assert params_mixing_matrix_sample.shape[0] == self.number_frequencies-len_pos_special_freqs
                assert params_mixing_matrix_sample.shape[1] == self.number_correlations-1
                assert len(params_mixing_matrix_sample.shape) == 2
                # Recording of the samples
            
                
                mixing_matrix_obj.update_params(params_mixing_matrix_sample)
                mixing_matrix_sampled = np.copy(mixing_matrix_obj.get_B())

                assert mixing_matrix_sampled.shape[0] == self.number_frequencies
                assert mixing_matrix_sampled.shape[1] == self.number_components
            all_params_mixing_matrix_samples[iteration+1,...] = params_mixing_matrix_sample


            # Few tests to verify everything's fine
            all_eigenvalues = np.linalg.eigh(red_cov_matrix_sample[self.lmin:])[0]
            assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_cov_matrix_sample[self.lmin:])[0])>10**(-15)]>0)

            # if iteration%50 == 0:
            #     print("### Iteration n°", iteration, flush=True)
            print("### Iteration n°", iteration, flush=True)

        time_full_chain = (time.time()-time_start_sampling)/60
        print("End of iterations in {} minutes, saving all files !".format(time_full_chain), flush=True)

        self.dict_last_samples = {'input_freq_maps':input_freq_maps, 'c_ell_approx':c_ell_approx, 
                             'CMB_covariance':all_cell_samples[-1,...], 
                             'init_params_mixing_matrix':all_params_mixing_matrix_samples[-1,...],
                             'initial_guess_r':all_r_samples[-1], 
                             'initial_wiener_filter_term':all_s_c_WF_maps[-1,...], 
                             'initial_fluctuation_maps':all_s_c_fluct_maps[-1,...],
                             'theoretical_r0_total':theoretical_r0_total, 
                             'theoretical_r1_tensor':theoretical_r1_tensor}

        return all_eta, all_s_c_WF_maps, all_s_c_fluct_maps, all_cell_samples, all_r_samples, all_params_mixing_matrix_samples

    def restart_sampling(self):
        print("Changing seed for new computations", flush=True)
        self.seed = self.seed + self.number_iterations_done
        return self.perform_sampling(self.dict_last_samples)

def create_object_from_dict(new_number_iterations_sampling, dict_object):
    new_object =  MICMAC_Sampler(**dict_object)
    new_object.number_iterations_done = new_object.number_iterations_sampling
    new_object.number_iterations_sampling = new_number_iterations_sampling
    return new_object
