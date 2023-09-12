import os, sys, time
import numpy as np
import healpy as hp
import scipy
# import mappraiser_wrapper_WF as mappraiser
from .tools import *
from .proba_functions import *

def get_inverse_wishart_sampling_from_c_ells(sigma_ell, q_prior=0, l_min=0, option_ell_2=1):
    """ Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
            1. Sample n = 2*ell - p + 2*q_prior independent Gaussian vectors with covariance (sigma_ell)^{-1}
            2. Compute their outer product to form a matrix of dimension n_stokes*n_stokes ; which gives us a sample following the Wishart distribution
            3. Invert this matrix to obtain the final result : a matrix sample following an inverse Wishart distribution

        Also assumes the monopole and dipole to be 0

        Parameters
        ----------
        sigma_ell : initial power spectrum which will define the parameter matrix of the inverse Wishart distribution ; must be of dimension [number_correlations, lmax+1]
        
        q_prior : choice of prior for the distribution : 0 means uniform prior ; 1 means Jeffrey prior
        
        lmin : minimum multipole to be considered, default 0

        option_ell_2 : option to choose how to sample ell=2, which is not defined by inverse Wishart distribution if nstokes=3 ; ignored if lmin>2 and/or nstokes<3
                       case 0 : not caring about the fact the inverse Wishart distribution is not defined and still sample as if it was 
                       case 1 : sample ell=2 with Jeffrey prior (only for ell=2)
                       case 2 : sample ell=2 by sampling separately the TE and B blocks respectively, assumes TB and EB to be 0

        Returns
        -------
        Matrices following an inverse Wishart distribution, of dimensions [lmin:lmax, nstokes, nstokes]
    """

    if len(sigma_ell.shape) == 1:
        nstokes == 1
        lmax = len(sigma_ell) - 1
    elif sigma_ell.shape[0] == 6:
        nstokes = 3
        lmax = sigma_ell.shape[1] - 1
    elif sigma_ell.shape[0] == 3:
        nstokes = 2
        lmax = sigma_ell.shape[1] - 1

    for i in range(nstokes):
        sigma_ell[i] *= 2*np.arange(lmax+1) + 1

    lmin = l_min

    invert_parameter_Wishart = np.linalg.pinv(get_reduced_matrix_from_c_ell(sigma_ell))
    
    assert invert_parameter_Wishart.shape[0] == lmax + 1 #- lmin
    sampling_Wishart = np.zeros_like(invert_parameter_Wishart)

    assert (option_ell_2 == 0) or (option_ell_2 == 1) or (option_ell_2 == 2)
    # Option sampling without caring about inverse Wishart not defined
    ell_2 = 2
    if l_min <= 2 and (2*ell_2 + 1 - 2*nstokes + 2*q_prior <= 0):
        # 2*ell_2 + 1 - 2*nstokes + 2*q_prior <= 0) correspond to the definition condition of the inverse Wishart distribution

        # Option sampling with brute force inverse Wishart
        if option_ell_2 == -1:
            print("~~Not caring about inverse Wishart distribution not defined !", flush=True)
            mean = np.zeros(nstokes)
            sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*q_prior))
            sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)
            
        # Option sampling with Jeffrey prior
        elif option_ell_2 == 0:
            Jeffrey_prior = 1
            print("~~Applying Jeffry prior for ell=2 !", flush=True)
            mean = np.zeros(nstokes)
            sample_gaussian = np.random.multivariate_normal(mean, invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*Jeffrey_prior))
            sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)

        # Option sampling separately TE and B
        elif option_ell_2 == 1:
            print("~~Sampling separately TE and B for ell=2 !", flush=True)
            invert_parameter_Wishart_2 = np.zeros((nstokes,nstokes))
            reduced_matrix_2 = get_reduced_matrix_from_c_ell(sigma_ell)[ell_2]
            invert_parameter_Wishart_2[:nstokes-1, :nstokes-1] = np.linalg.pinv(reduced_matrix_2[:nstokes-1,:nstokes-1])
            invert_parameter_Wishart_2[nstokes-1, nstokes-1] = 1/reduced_matrix_2[nstokes-1,nstokes-1]
            sample_gaussian_TE = np.random.multivariate_normal(np.zeros(nstokes-1), invert_parameter_Wishart_2[:nstokes-1, :nstokes-1], size=(2*ell_2 - (nstokes-1)))
            sample_gaussian_B = np.random.normal(loc=0, scale=invert_parameter_Wishart_2[nstokes-1, nstokes-1], size=(2*ell_2 - 1))
            sampling_Wishart[ell_2][:nstokes-1,:nstokes-1] = np.dot(sample_gaussian_TE.T,sample_gaussian_TE)
            sampling_Wishart[ell_2][nstokes-1,nstokes-1] = np.dot(sample_gaussian_B.T,sample_gaussian_B)
        
        lmin = 3
    for ell in range(max(lmin,2),lmax+1):
        sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell], size=(2*ell - nstokes + 2*q_prior))
        sampling_Wishart[ell] = np.dot(sample_gaussian.T,sample_gaussian)
    # sampling_Wishart[max(lmin,2):,...] = np.einsum('lkj,lkm->ljm',sample_gaussian,sample_gaussian)
    return np.linalg.pinv(sampling_Wishart)


def get_fluctuating_term_maps(param_dict, red_cov_matrix, red_inverse_noise, map_white_noise_xi, map_white_noise_chi, lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
    """ 
        Solve fluctuation term with formulation (C^-1 + N^-1) for the left member

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax
        
        red_cov_matrix : covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        red_inverse_noise : matrices of inverse noise in harmonic domain (yet), dimension [lmin:lmax, nstokes, nstokes]

        map_white_noise_xi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        map_white_noise_chi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        limit_iter_cg : maximum number of iterations for the CG, default 1000
        tolerance : CG tolerance, default 10**(-12)

        Returns
        -------
        Fluctuation maps [nstokes, npix]
    """

    assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin

    red_inverse_cov_matrix = np.linalg.pinv(red_cov_matrix)
    

    # Creation of the random maps
    if len(map_white_noise_xi) == 0:
        map_white_noise_xi = np.random.normal(loc=0, scale=1, size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_white_noise_chi) == 0:
        map_white_noise_chi = np.random.normal(loc=0, scale=1, size=(param_dict["nstokes"],12*param_dict["nside"]**2))

    # Computation of the right side member of the CG
    red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(red_inverse_cov_matrix)
    red_inv_noise_sqrt = get_sqrt_reduced_matrix_from_matrix(red_inverse_noise)
    right_member_1 = maps_x_reduced_matrix_generalized_sqrt(map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_inv_cov_sqrt, lmin=lmin, n_iter=n_iter)
    right_member_2 = maps_x_reduced_matrix_generalized_sqrt(map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_inv_noise_sqrt, lmin=lmin, n_iter=n_iter)

    right_member = (right_member_1 + right_member_2).ravel()
    
    # Computation of the left side member of the CG
    new_matrix_cov = red_inverse_cov_matrix + red_inverse_noise
    new_matrix_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(new_matrix_cov)
    func_left_term = lambda x : (maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_matrix_cov_sqrt, lmin=lmin, n_iter=n_iter)).ravel()

    # Initial guess for the CG
    initial_guess = np.zeros_like(map_white_noise_xi)

    # Actual start of the CG
    fluctuating_map, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    print("CG-Python-0 Fluct sqrt finished in ", number_iterations, "iterations for fluctuating term !!")    

    if exit_code != 0:
        print("CG didn't converge with generalized_CG for fluctuating term ! Exitcode :", exit_code, flush=True)
    return fluctuating_map.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))



def solve_generalized_wiener_filter_term(param_dict, data_var, red_cov_matrix, red_inverse_noise, lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
    """ 
        Solve Wiener filter term with formulation (1 + C^1/2 N^-1 C^1/2) for the left member

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax

        data_var : data maps, for Wiener filter CG ; dimensions [nstokes, npix]
        
        red_cov_matrix : covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        red_inverse_noise : matrices of inverse noise in harmonic domain (yet), dimension [lmin:lmax, nstokes, nstokes]

        map_white_noise_xi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        map_white_noise_chi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        limit_iter_cg : maximum number of iterations for the CG, default 1000
        tolerance : CG tolerance, default 10**(-12)

        Returns
        -------
        Wiener filter maps [nstokes, npix]
    """

    assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin
    if param_dict['nstokes'] != 1:
        assert data_var.shape[0] == param_dict['nstokes']
    

    # Computation of the right side member of the CG
    red_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(red_cov_matrix)
    # new_right_member_mat = np.zeros_like(red_cov_matrix)
    new_right_member_mat = np.einsum('ljk,lkm->ljm', red_cov_sqrt,red_inverse_noise)
    right_member = (maps_x_reduced_matrix_generalized_sqrt(np.copy(data_var).reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_right_member_mat, lmin=lmin, n_iter=n_iter)).ravel()    

    # Computation of the left side member of the CG
    new_matrix_cov = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)
    func_left_term = lambda x : (maps_x_reduced_matrix_generalized_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_matrix_cov, lmin=lmin, n_iter=n_iter)).ravel()

    # Initial guess for the CG
    initial_guess = np.zeros_like(data_var)
    # Actual start of the CG
    wiener_filter_term_z, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    print("CG-Python-1 WF sqrt finished in ", number_iterations, "iterations !!")

    # Change of variable to get the correct Wiener Filter term
    wiener_filter_term = maps_x_reduced_matrix_generalized_sqrt(wiener_filter_term_z.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_sqrt, lmin=lmin, n_iter=n_iter)

    if exit_code != 0:
        print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
    return wiener_filter_term.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))


class Gibbs_Sampling(object):
    def __init__(self, nside, lmax, nstokes, lmin=0, n_iter=8, number_iterations_sampling=1000, limit_iter_cg=1000, tolerance_fluctuation=10**(-4), tolerance_PCG=10**(-12), prior=0, option_ell_2=1):
        """ Gibbs sampling object
        """

        # # Package params
        # assert (language == 'C') or (language == 'Python')
        # self.language=language

        # Problem parameters
        self.nside = nside
        self.lmax = lmax
        self.nstokes = nstokes
        self.lmin = lmin
        self.n_iter = n_iter # Number of iterations for Python estimation of alms

        # CG parameters
        self.limit_iter_cg = limit_iter_cg # Maximum number of iterations for the different CGs
        self.tolerance_PCG = tolerance_PCG # Tolerance for the Wiener filter CG
        self.tolerance_fluctuation = tolerance_fluctuation # Tolerance for the fluctuation CG

        # Sampling parameters
        self.number_iterations_sampling = number_iterations_sampling # Maximum number of iterations for the sampling
        self.prior = prior
        # For the prior :
            # 0 : uniform prior
            # 1 : Jeffrey prior
        self.option_ell_2 = option_ell_2
        # For the option_ell_2 :
            # 0 : Wishart classical (despite the fact it's not defined for ell=2 and nstokes=3)
            # 1 : Jeffrey prior (with a Jeffrey prior only for ell=2)
            # 2 : Sampling separately the TE and B blocks respectively, only for ell=2

    @property
    def npix(self):
        return 12*self.nside**2

    @property
    def number_correlations(self):
        """ Maximum number of correlations depending of the number of Stokes parameters : 
            6 (TT,EE,BB,TE,EB,TB) for 3 Stokes parameters ; 3 (EE,BB,EB) for 2 Stokes parameters ; 1 (TT) for 1 Stokes parameter"""
        return int(np.ceil(self.nstokes**2/2) + np.floor(self.nstokes/2))
    
    def constrained_map_realization(self, initial_map, red_covariance_matrix, red_inverse_noise):
        """ Constrained map realization step, given the data """
        
        assert red_covariance_matrix.shape[0] == self.lmax + 1 - self.lmin
        assert red_inverse_noise.shape[0] == self.lmax + 1 - self.lmin
        if self.nstokes != 1:
            assert initial_map.shape[0] == self.nstokes
        
        param_dict = {'nside':self.nside, 'lmax':self.lmax, 'nstokes':self.nstokes, 'number_correlations':self.number_correlations}
        
        initial_guess = np.zeros((self.nstokes, self.npix))

        map_white_noise_xi = []
        map_white_noise_chi = []

        fluctuating_map = get_fluctuating_term_maps(param_dict, red_covariance_matrix, red_inverse_noise, map_white_noise_xi, map_white_noise_chi, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_fluctuation)
        
        wiener_filter_term = solve_generalized_wiener_filter_term(param_dict, initial_map, red_covariance_matrix, red_inverse_noise, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_PCG)

        return wiener_filter_term + fluctuating_map
    
    def sample_covariance(self, pixel_maps):
        """ Power spectrum sampling, given the sampled maps """
        c_ells_Wishart = get_cell_from_map(pixel_maps, lmax=self.lmax, n_iter=self.n_iter)
        return get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart, lmax=self.lmax, q_prior=self.prior, l_min=self.lmin, option_ell_2=self.option_ell_2)#[self.lmin:,...]

    def perform_sampling(self, initial_map, c_ells_array, red_inverse_noise):
        """ Perform sampling steps with :
                1. The constrained realization step
                2. The c_ell sampling
        """

        # Preparation of the initial guess
        c_ell_sampled = np.copy(c_ells_array)
        # Preparation of the input map data
        pixel_maps_sampled = np.copy(initial_map)

        if self.nstokes != 1:
            assert initial_map.shape[0] == self.nstokes

        all_maps = np.zeros((self.number_iterations_sampling+1, self.nstokes, self.npix))
        all_samples = np.zeros((self.number_iterations_sampling+1, self.number_correlations, self.lmax+1))

        all_maps[0,...] = pixel_maps_sampled
        all_samples[0,...] = c_ell_sampled

        if red_inverse_noise.shape[0] == self.lmax+1:
            red_inverse_noise = red_inverse_noise[self.lmin:,...]

        for iteration in range(self.number_iterations_sampling):
            print("### Start Iteration n°", iteration, flush=True)

            red_covariance_matrix = get_reduced_matrix_from_c_ell(c_ell_sampled)[self.lmin:,...]

            # Constrained realization step
            pixel_maps_sampled = self.constrained_map_realization(initial_map, red_covariance_matrix, red_inverse_noise)
            
            # C_ell sampling step
            red_cov_mat_sampled = self.sample_covariance(pixel_maps_sampled)

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
