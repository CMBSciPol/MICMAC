import os, sys, time
import numpy as np
import healpy as hp
import scipy

from .tools import *
from .algorithm_toolbox import *
from .proba_functions import *
from .Sampling_toolbox import *
from .mixingmatrix import *
from noisecovar import *

class Non_parametric_Likelihood_Sampling(object):
    def __init__(self, nside, lmax, nstokes, number_frequencies, number_components, lmin=0, n_iter=8, number_iterations_sampling=1000, limit_iter_cg=1000, tolerance_CG=10**(-12), option_ell_2=2, number_walkers=1, limit_steps_sampler_mixing_matrix=1000):
        """ Non parametric likelihood sampling object
        """

        # Problem parameters
        self.nside = nside
        self.lmax = lmax
        self.nstokes = nstokes
        self.lmin = lmin
        self.n_iter = n_iter # Number of iterations for Python estimation of alms
        self.number_frequencies = number_frequencies
        self.number_components = number_components

        # CG parameters
        self.limit_iter_cg = limit_iter_cg # Maximum number of iterations for the different CGs
        self.tolerance_CG = tolerance_CG # Tolerance for the different CGs

        # Sampling parameters
        self.number_iterations_sampling = number_iterations_sampling # Maximum number of iterations for the sampling
        self.option_ell_2 = option_ell_2
        # For the option_ell_2 :
            # 0 : Wishart classical (despite the fact it's not defined for ell=2 and nstokes=3)
            # 1 : Jeffrey prior (with a Jeffrey prior only for ell=2)
            # 2 : Sampling separately the TE and B blocks respectively, only for ell=2
        self.number_walkers = number_walkers # Number of walkers for the MCMC to sample the mixing matrix
        self.limit_steps_sampler_mixing_matrix = limit_steps_sampler_mixing_matrix # Maximum number of steps for the MCMC to sample the mixing matrix

    @property
    def npix(self):
        return 12*self.nside**2

    @property
    def number_correlations(self):
        """ Maximum number of correlations depending of the number of Stokes parameters : 
            6 (TT,EE,BB,TE,EB,TB) for 3 Stokes parameters ; 3 (EE,BB,EB) for 2 Stokes parameters ; 1 (TT) for 1 Stokes parameter"""
        return int(np.ceil(self.nstokes**2/2) + np.floor(self.nstokes/2))

    def sample_covariance(self, pixel_maps):
        """ Power spectrum sampling, given the sampled maps, following inverse Wishart distribution """
        c_ells_Wishart = get_cell_from_map(pixel_maps, lmax=self.lmax, n_iter=self.n_iter)
        return get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart, l_min=self.lmin, option_ell_2=self.option_ell_2)#[self.lmin:,...]


    def perform_sampling(self, initial_freq_maps, tot_cov_first_guess, depth_p, c_ell_approx, init_params_mixing_matrix, pos_special_freqs):
        """ Perform sampling steps with :
                1. A CG on variable eta for (S_approx + mixed_noise) eta = S_approx^(1/2) x + E^t (B^t N^{-1} B)^{-1} E noise^(1/2) y
                2. A CG for the Wiener filter variable s_c : (s_c - s_c,ML)^t (S_c + E^t (B^t N^{-1} B)^{-1} E) (s_c - s_c,ML)
                3. The c_ell sampling assuming inverse Wishart distribution
                4. Mixing matrix B_f sampling with : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t (S_{approx} + E^t (B^T N^{-1} B)^{-1} E) eta
            
            Parameters
            ----------
            initial_freq_maps : data of initial frequency maps, dimensions [frequencies, nstokes, npix]
            
            tot_cov_first_guess : total covariance first guess, composed of all c_ell correlations in order (for polarization [EE, BB, EB])

            depth_p: depth for the noise covariance calculation, in uK.arcmin

            c_ell_approx : 

            init_params_mixing_matrix : 
            
            pos_special_freqs :

            Returns
            -------
        """

        # Parameters initialization
        freq_inverse_noise = get_noise_covar(depth_p, self.nside)
        red_cov_approx_matrix = get_reduced_matrix_from_c_ell(c_ell_approx)[self.lmin:,...]


        # Initialization of CG maps to 0 for first iteration
        eta_maps = np.zeros((self.nstokes, self.npix)) # copy initial map better ?

        # Preparation of the initial guess
        c_ell_sampled = np.copy(tot_cov_first_guess)

        # Preparation of the input map data
        pixel_maps_sampled = np.copy(initial_freq_maps)

        # Preparation of covariance matrix
        red_covariance_matrix_sampled = get_reduced_matrix_from_c_ell(c_ell_sampled)[self.lmin:,...]

        # Preparation of the mixing matrix object
        Mixingmatrix_obj = MixingMatrix(self.number_frequencies, self.number_components, init_params_mixing_matrix, pos_special_freqs)
        mixing_matrix_sampled = Mixingmatrix_obj.get_B()


        if self.nstokes != 1:
            assert initial_freq_maps.shape[0] == self.nstokes
        
        param_dict = {'nside':self.nside, 'lmax':self.lmax, 'nstokes':self.nstokes, 'number_correlations':self.number_correlations}
        

        all_eta = np.zeros((self.number_iterations_sampling+1, self.nstokes, self.npix))
        all_maps = np.zeros((self.number_iterations_sampling+1, self.nstokes, self.npix))
        all_cell_samples = np.zeros((self.number_iterations_sampling+1, self.number_correlations, self.lmax+1))
        all_mixing_matrix_samples = np.zeros((self.number_iterations_sampling+1, self.number_correlations, self.number_frequencies))

        # Initial values
        all_maps[0,...] = pixel_maps_sampled
        all_cell_samples[0,...] = c_ell_sampled
        all_mixing_matrix_samples[0,...] = mixing_matrix_sampled

        if self.nstokes != 1:
                assert initial_freq_maps.shape[0] == self.number_frequencies
                assert initial_freq_maps.shape[1] == self.nstokes
                assert initial_freq_maps.shape[2] == self.npix

        assert freq_inverse_noise.shape[0] == self.number_frequencies
        assert freq_inverse_noise.shape[1] == self.number_frequencies

        

        for iteration in range(self.number_iterations_sampling):
            print("### Start Iteration n°", iteration, flush=True)

            # Application of new mixing matrix
            cp_cp_noise = get_inv_BtinvNB(freq_inverse_noise, mixing_matrix_sampled)
            cp_freq_inv_noise_sqrt = get_BtinvN(np.sqrt(freq_inverse_noise), mixing_matrix_sampled)
            ML_initial_data_maps = np.einsum('cd,df,fsp->s', cp_cp_noise, get_BtinvN(freq_inverse_noise, mixing_matrix_sampled), initial_freq_maps)[0] # Computation of E^t (B^t N^{-1} B)^{-1} B^t N^{-1} d


            assert eta_maps.shape[0] == self.nstokes
            assert eta_maps.shape[1] == self.npix
            assert CMB_pixel_maps_sampled.shape[0] == self.nstokes
            assert CMB_pixel_maps_sampled.shape[1] == self.npix
            assert red_covariance_matrix_sampled.shape[0] == self.lmax + 1 - self.lmin
            assert mixing_matrix_sampled.shape[0] == self.number_components
            assert mixing_matrix_sampled.shape[1] == self.number_frequencies
                
            
            # Sampling step 1 : Solve CG for eta term with formulation : (S_approx + mixed_noise) eta = S_approx^(-1/2) x + mixed_noise noise^(1/2) y
            map_random_x = []
            map_random_y = []
            eta_maps = get_sampling_eta(param_dict, red_cov_approx_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, initial_guess=np.copy(eta_maps), lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_CG)

            # Sampling step 2 : sampling of Gaussian variable s_c with mean ML_initial_data_maps and variance (S_c + E^t (B^t N^{-1} B)^{-1} E)
            map_random_xi = []
            map_random_chi = []
            CMB_pixel_maps_sampled = get_gaussian_sample_maps(param_dict, ML_initial_data_maps, red_covariance_matrix_sampled, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_xi=map_random_xi, map_random_realization_chi=map_random_chi, lmin=self.lmin, n_iter=self.n_iter)

            # Application of new gaussian maps
            extended_CMB_maps = np.zeros((self.number_components, self.nstokes, self.npix))
            extended_CMB_maps[0] = CMB_pixel_maps_sampled
            full_data_without_CMB = initial_freq_maps - np.einsum('fc,csp->fsp',mixing_matrix_sampled, extended_CMB_maps)
            assert full_data_without_CMB.shape[0] == self.number_frequencies
            assert full_data_without_CMB.shape[1] == self.nstokes
            assert full_data_without_CMB.shape[2] == self.npix
            
            # Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
            red_covariance_matrix_sampled = self.sample_covariance(pixel_maps_sampled)

            # Sampling step 4
            initial_guess_mixing_matrix = []
            sample_fg_mixing_matrix = sample_mixing_matrix_term(param_dict, full_data_without_CMB, eta_maps, red_cov_approx_matrix, freq_inverse_noise, Mixingmatrix_obj, initial_guess_mixing_matrix=mixing_matrix_sampled, lmin=self.lmin, n_iter=self.n_iter, n_walkers=self.number_walkers, limit_steps_sampler_mixing_matrix=self.limit_steps_sampler_mixing_matrix)
            mixing_matrix_sampled[:,1:] = sample_fg_mixing_matrix

            # Few tests to verify everything's fine
            all_eigenvalues = np.linalg.eigh(red_covariance_matrix_sampled[self.lmin:])[0]
            assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_covariance_matrix_sampled[self.lmin:])[0])>10**(-15)]>0)

            
            # Recording of the samples
            all_eta[iteration+1,...] = eta_maps
            all_maps[iteration+1,...] = pixel_maps_sampled
            all_cell_samples[iteration+1,...] = get_c_ells_from_red_covariance_matrix(red_covariance_matrix_sampled)
            all_mixing_matrix_samples[iteration+1,...] = mixing_matrix_sampled

            if iteration%50 == 0:
                print("### Iteration n°", iteration, flush=True)
        return all_eta, all_maps, all_cell_samples, all_mixing_matrix_samples
