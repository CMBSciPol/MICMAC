import os, sys, time
import numpy as np
import healpy as hp
import scipy

from .tools import *
from .algorithm_toolbox import *
from .proba_functions import *
from .Sampling_toolbox import *

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


    def perform_sampling(self, initial_freq_maps, tot_cov_first_guess, red_inverse_noise, red_cov_approx_matrix, mixing_matrix):
        """ Perform sampling steps with :
                1. A CG on variable eta for (S_approx + mixed_noise) eta = S_approx^(-1/2) x + mixed_noise noise^(1/2) y
                2. A CG for the Wiener filter variable s_c : (s_c - s_c,ML)^t (S_c + mixed_noise) (s_c - s_c,ML)
                3. The c_ell sampling assuming inverse Wishart distribution
                4. Mixing matrix B_f sampling with : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t (S_{approx} + E^t (B^T N^{-1} B)^{-1} E) eta
        """

        # Preparation of the initial guess
        c_ell_sampled = np.copy(tot_cov_first_guess)
        # Preparation of the input map data
        pixel_maps_sampled = np.copy(initial_freq_maps)

        if self.nstokes != 1:
            assert initial_freq_maps.shape[0] == self.nstokes
        
        param_dict = {'nside':self.nside, 'lmax':self.lmax, 'nstokes':self.nstokes, 'number_correlations':self.number_correlations}
        

        all_maps = np.zeros((self.number_iterations_sampling+1, self.nstokes, self.npix))
        all_samples = np.zeros((self.number_iterations_sampling+1, self.number_correlations, self.lmax+1))

        # Initial values
        all_maps[0,...] = pixel_maps_sampled
        all_samples[0,...] = c_ell_sampled

        # Initialization of eta and WF maps to 0 for first iteration
        eta_maps = np.zeros_like(initial_freq_maps) # copy initial map better ?
        wiener_filter_term = np.zeros_like(initial_freq_maps) # copy initial map better ?

        if red_inverse_noise.shape[0] == self.lmax+1:
            red_inverse_noise = red_inverse_noise[self.lmin:,...]

        
        
        for iteration in range(self.number_iterations_sampling):
            print("### Start Iteration n°", iteration, flush=True)

            red_covariance_matrix = get_reduced_matrix_from_c_ell(c_ell_sampled)[self.lmin:,...]

            # Initialization mixing matrix


                
            assert red_covariance_matrix.shape[0] == self.lmax + 1 - self.lmin
            assert red_inverse_noise.shape[0] == self.lmax + 1 - self.lmin
            if self.nstokes != 1:
                assert initial_freq_maps.shape[0] == self.nstokes
            
            
            # Sampling step 1 : Solve CG for eta term with formulation : (S_approx + mixed_noise) eta = S_approx^(-1/2) x + mixed_noise noise^(1/2) y
            map_random_x = []
            map_random_y = []
            eta_maps = get_sampling_eta(param_dict, red_cov_approx_matrix, red_inverse_noise, red_mixed_noise, map_random_x=map_random_x, map_random_y=map_random_y, initial_guess=np.copy(eta_maps), lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_CG)

            # Sampling step 2 : sampling of s_c with Wiener filter
            wiener_filter_term = solve_generalized_wiener_filter_term(param_dict, ML_initial_data, red_covariance_matrix, red_inverse_noise, initial_guess=np.copy(wiener_filter_term), lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_CG)

    
            # Sampling step 3 : c_ell sampling assuming inverse Wishart distribution
            red_cov_mat_sampled = self.sample_covariance(pixel_maps_sampled)

            # Sampling step 4
            initial_guess_mixing_matrix = []
            mixing_matrix_sampled = sample_mixing_matrix_term(param_dict, full_data, transformed_data, red_cov_approx_matrix, red_inverse_mixing_noise, initial_guess_mixing_matrix=initial_guess_mixing_matrix, lmin=self.lmin, n_iter=self.n_iter, n_walkers=self.number_walkers, limit_steps_sampler_mixing_matrix=self.limit_steps_sampler_mixing_matrix)


            # Few tests to verify everything's fine
            all_eigenvalues = np.linalg.eigh(red_cov_mat_sampled[self.lmin:])[0]
            assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_cov_mat_sampled[self.lmin:])[0])>10**(-15)]>0)

            # Preparation of next step
            c_ell_sampled = get_c_ells_from_red_covariance_matrix(red_cov_mat_sampled)
            
            # Recording of the samples
            all_maps[iteration+1,...] = pixel_maps_sampled
            all_samples[iteration+1,...] = c_ell_sampled

            if iteration%50 == 0:
                print("### Iteration n°", iteration, flush=True)
        return all_maps, all_samples
