import os, sys, time
import numpy as np
import healpy as hp
import emcee
from .tools import *
from .algorithm_toolbox import *
from .proba_functions import *


def get_sampling_eta(param_dict, red_cov_approx_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_x=[], map_random_y=[], initial_guess=[], lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
    """ Solve sampling step 1 : sampling eta
        Solve CG for eta term with formulation : (S_approx + mixed_noise) eta = S_approx^(-1/2) x + mixed_noise noise^(1/2) y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax
        
        red_cov_approx_matrix : correction covariance matrice (S_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        cp_cp_noise : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        cp_freq_inv_noise_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

        map_random_x : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
        map_random_y : set of maps 0 with mean and variance 1/(pixel_size**2), which will be used to compute eta, default [] and it will be computed by the code ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        limit_iter_cg : maximum number of iterations for the CG, default 1000
        tolerance : CG tolerance, default 10**(-12)

        initial_guess : initial guess for the CG, default [] (which is a covnention for its initialization to 0)

        Returns
        -------
        eta maps [nstokes, npix]
    """

    assert red_cov_approx_matrix.shape[0] == param_dict['lmax'] + 1 - lmin

    # Creation of the random maps if they are not given
    if len(map_random_x) == 0:
        print("Recalculating xi !")
        map_random_x = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_random_y) == 0:
        print("Recalculating chi !")
        map_random_y = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))

    # Computation of the right hand side member of the CG
    red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix(red_cov_approx_matrix)

    # First right member : S_approx^{-1/2} x
    right_member_1 = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix_sqrt, lmin=lmin, n_iter=n_iter)

    # Second right member : E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2}
    right_member_2 = np.einsum('kc,cf,fsp', cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_y)[0] # Selecting CMB component of the random variable
    right_member_2 = get_band_limited_map(right_member_2, lmin=lmin, lmax=param_dict["lmax"])

    right_member = (right_member_1 + right_member_2).ravel()


    # Computation of the left hand side member of the CG
    first_term_left = lambda x : (maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix, lmin=lmin, n_iter=n_iter)).ravel()

    cmb_noise_map = cp_cp_noise[0,0]
    second_term_left = lambda x : (cmb_noise_map*(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)))).ravel()

    func_left_term = lambda x : first_term_left(x) + second_term_left(x)

    # Initial guess for the CG
    if len(initial_guess) == 0:
        print("Defining initial guess to be 0 for eta sampling")
        initial_guess = np.zeros_like(map_random_x)

    # Actual start of the CG
    eta_maps, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    print("CG-Python eta sampling finished in ", number_iterations, "iterations !!")    

    if exit_code != 0:
        print("CG didn't converge with generalized_CG for eta sampling ! Exitcode :", exit_code, flush=True)
    return eta_maps.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))

def get_gaussian_variance_maps(param_dict, red_cov_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_xi=[], map_random_realization_chi=[], lmin=0, n_iter=8):
    """ 
        Solve variance of sampling step 2 : sampling a Gaussian variable with variance (S_c + E^t (B^t N^{-1} B)^{-1} E)
        Obtain sample from Gaussian distribution : (s_c)^t (S_c + E^t (B^t N^{-1} B)^{-1} E)^{-1} (s_c)

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax
        
        red_cov_matrix : CMB covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        cp_cp_noise : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        cp_freq_inv_noise_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

        map_random_realization_xi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        map_random_realization_chi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        Returns
        -------
        CMB Maps with mean 0 and variance  (S_c + E^t (B^t N^{-1} B)^{-1} E), dimensions [nstokes, npix]
    """

    assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin

    # Creation of the random maps
    if len(map_random_realization_xi) == 0:
        print("Recalculating xi !")
        map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_random_realization_chi) == 0:
        print("Recalculating chi !")
        map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))

    # Computation of the right side member of the CG
    red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(np.linalg.pinv(red_cov_matrix))

    first_term = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_realization_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_inv_cov_sqrt, lmin=lmin, n_iter=n_iter)

    second_term = np.einsum('kc,cf,fsp', cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_chi)[0] # Selecting CMB component of the random variable

    # Make second term band limited ?
    second_term = get_band_limited_map(second_term, lmin=lmin, lmax=param_dict["lmax"])

    return first_term + second_term


def get_gaussian_sample_maps(param_dict, mean_gaussian_sample_maps, red_cov_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_xi=[], map_random_realization_chi=[], lmin=0, n_iter=8):
    """ 
        Solve complete sampling step 2 : sampling a Gaussian variable with mean s_c_ML and variance (S_c + E^t (B^t N^{-1} B)^{-1} E)
        Obtain sample from Gaussian distribution : (s_c - s_c,ML)^t (S_c + E^t (B^t N^{-1} B)^{-1} E)^{-1} (s_c - s_c,ML) 

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax
        
        red_cov_matrix : CMB covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        cp_cp_noise : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        cp_freq_inv_noise_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

        map_random_realization_xi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        map_random_realization_chi : set of maps 0 with mean and variance 1, which will be used to compute the fluctuation term ; dimension [nstokes, npix]
        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        Returns
        -------
        CMB Maps with mean 0 and variance  (S_c + E^t (B^t N^{-1} B)^{-1} E), dimensions [nstokes, npix]
    """

    variance_sample = get_gaussian_variance_maps(param_dict, red_cov_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_xi=map_random_realization_xi, map_random_realization_chi=map_random_realization_chi, lmin=lmin, n_iter=n_iter)

    return mean_gaussian_sample_maps +  variance_sample


def get_inverse_wishart_sampling_from_c_ells(sigma_ell, q_prior=0, l_min=0, option_ell_2=1):
    """ Solve sampling step 3 : inverse Wishart distribution with S_c
        Compute a matrix sample following an inverse Wishart distribution. The 3 steps follow Gupta & Nagar (2000) :
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



def get_conditional_proba_mixing_matrix(variable_mixing_matrix, full_data_without_CMB, eta, freq_freq_inverse_noise, red_cov_approx_matrix, param_dict, lmin, n_iter):
    """ Get conditional probability of mixing matrix

        cp_cp_noise : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        cp_freq_inv_noise : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]
    """

    nstokes = eta.shape[0]
    npix = eta.shape[1]

    mixing_matrix_to_find = variable_mixing_matrix.reshape((param_dict["number_components", "number_frequencies"]))

    cp_cp_noise = get_cp_cp_noise(freq_freq_inverse_noise, variable_mixing_matrix)
    cp_freq_inv_noise = get_cp_freq_noise(freq_freq_inverse_noise, variable_mixing_matrix)

    
    # Building the first term
    mixing_matrix_fg_only = variable_mixing_matrix[1:,1:]
    full_data_without_CMB_with_noise = np.einsum('cf,fsp->csp', cp_freq_inv_noise,full_data_without_CMB)
    
    first_term_complete = np.einsum('psc,cm,msp ', full_data_without_CMB_with_noise.T, cp_cp_noise, full_data_without_CMB_with_noise)

    # Building the second term term \eta^t (S_approx + E^t (B^t N^{-1} B)^{-1} E) \eta
    second_term_1 = maps_x_reduced_matrix_generalized_sqrt_sqrt(eta.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix, lmin=lmin, n_iter=n_iter)

    cmb_noise_map = cp_cp_noise[0,0]
    second_term_2 = cmb_noise_map*(eta.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)))
    
    second_term_complete = np.einsum('sk,sk',eta, second_term_1 + second_term_2)

    return first_term_complete + second_term_complete



def sample_mixing_matrix_term(param_dict, full_data, transformed_data, red_cov_approx_matrix, red_inverse_mixing_noise, initial_guess_mixing_matrix=[], lmin=0, n_iter=8, n_walkers=8, number_steps_sampler=1000):
    """ Solve sampling step 4 : sampling B_f
        Sample mixing matrix with formualtion : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t (S_{approx} + E^t (B^T N^{-1} B)^{-1} E) eta

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax

        full_data : full data maps, for Wiener filter CG ; dimensions [nstokes, npix]
        
        red_cov_matrix : covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        red_inverse_noise : matrices of inverse noise in harmonic domain (yet), dimension [lmin:lmax, nstokes, nstokes]

        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        Returns
        -------
        Mixing matrix sampled
    """

    # assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    # assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin
    # if param_dict['nstokes'] != 1:
    #     assert data_variable.shape[0] == param_dict['nstokes']
    

    dimensions_mixing_matrix = param_dict['number_frequencies']*param_dict['number_components']

    sampler_mixing_matrix = emcee.EnsembleSampler(n_walkers, dimensions_mixing_matrix, get_conditional_proba_mixing_matrix, args=[full_data_without_CMB, eta, inverse_noise, red_cov_approx_matrix, param_dict, lmin, n_iter])

    full_initial_guess = np.repeat(initial_guess_mixing_matrix.ravel(), n_walkers)
    
    sampler_mixing_matrix.run_mcmc(full_initial_guess, number_steps_sampler)

    return sampler_mixing_matrix.get_chain().reshape((n_walkers, param_dict['number_frequencies'],param_dict['number_components']))
