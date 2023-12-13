import os, sys, time
import numpy as np
import healpy as hp
import scipy.stats
import emcee
from .tools import *
from .algorithm_toolbox import *
from .proba_functions import *
from .mixingmatrix import *
from .noisecovar import *


def get_sampling_eta_prime(param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=[], map_random_y=[], initial_guess=[], lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), suppress_low_modes=True):
    """ Solve sampling step 1 : sampling eta'
        Solve CG for eta term with formulation : eta' = C_approx^(1/2) x + (B^t N^{-1} B)^{-1} B^T N^{-1/2} y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
        
        red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        BtinvNB : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        # BtinvN_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

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
        print("Recalculating x !")
        map_random_x = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_random_y) == 0:
        print("Recalculating y !")
        map_random_y = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))

    # Computation of the right hand side member of the CG
    red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix(red_cov_approx_matrix)

    # First right member : C_approx^{1/2} x
    first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix_sqrt, lmin=lmin, n_iter=n_iter)

    # # Second right member : E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2}
    second_member = np.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_y)[0] # Selecting CMB component of the random variable

    map_solution = first_member + second_member
    if suppress_low_modes:
        covariance_unity = np.zeros((param_dict['lmax']+1,param_dict["nstokes"],param_dict["nstokes"]))
        covariance_unity[lmin:,...] = np.eye(param_dict["nstokes"])
        map_solution = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(map_solution), covariance_unity, lmin=0, n_iter=n_iter)

    return map_solution

def get_sampling_eta(param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=np.empty(0), map_random_y=np.empty(0), initial_guess=np.empty(0), lmin=0, n_iter=8, suppress_low_modes=True):
    """ Solve sampling step 1 : sampling eta'
        Solve CG for eta term with formulation : eta' = C_approx^(1/2) x + (B^t N^{-1} B)^{-1} B^T N^{-1/2} y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
        
        red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        BtinvNB : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        # BtinvN_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

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

    # assert red_cov_approx_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    eta_prime = get_sampling_eta_prime(param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=map_random_x, map_random_y=map_random_y, initial_guess=np.copy(initial_guess), lmin=lmin, n_iter=n_iter, suppress_low_modes=suppress_low_modes)
    
    # eta_prime_extended_frequencies = np.repeat(eta_prime, param_dict['number_frequencies']).reshape((param_dict['number_frequencies'],param_dict['nstokes'],12*param_dict['nside']**2),order='F')

    # Transform into eta maps by applying N_c^{-1/2} = N_c^{-1} N_c^{1/2} = (E^t (B^t N^{-1} B)^{-1} E)^{-1} E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2}
    # First applying N_c^{1/2}
    # first_part = np.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, eta_prime_extended_frequencies)
    # first_part = np.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, eta_prime_extended_frequencies)[0]
    # first_part = np.einsum('fc,c,sp->fsp', BtinvN_sqrt.T, BtinvNB.T[0,:], eta_prime/(BtinvNB[0,0]))

    N_c_sqrt = np.einsum('ck,kf->cf', BtinvNB, BtinvN_sqrt)[0,:]
    first_part = np.einsum('f,sp->fsp', N_c_sqrt, eta_prime/(BtinvNB[0,0]))


    # Then applying N_c^{-1}
    # return np.einsum('kc,csp->ksp', np.linalg.pinv(BtinvNB), first_part)[0]
    # return 1/(BtinvNB[0,0])*first_part
    return first_part

def get_sampling_eta_v2(param_dict, red_cov_approx_matrix, BtinvNB, BtinvN_sqrt, map_random_x=[], map_random_y=[], lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), suppress_low_modes=True):
    """ Solve sampling step 1 : sampling eta
        Solve CG for eta term with formulation : eta = C_approx^(1/2) (E (B^t N^{-1} B)^{-1} E^t)^{-1} C_approx^(1/2) x + y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
        
        red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        BtinvNB : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        # BtinvN_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

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
        print("Recalculating x !")
        map_random_x = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))
        # map_random_x = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_random_y) == 0:
        print("Recalculating y !")
        map_random_y = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))

    # Computation of the right hand side member of the CG
    red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix(red_cov_approx_matrix)

    # First right member : C_approx^(1/2) (E (B^t N^{-1} B)^{-1} E^t)^{-1} C_approx^(1/2) x    
    # first_member = map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))/(BtinvNB[0,0])
    first_member = np.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_x)[0]/BtinvNB[0,0] # Selecting CMB component of the random variable

    if suppress_low_modes:
        # covariance_unity = np.zeros((param_dict['lmax']+1,param_dict["nstokes"],param_dict["nstokes"]))
        # covariance_unity[lmin:,...] = np.eye(param_dict["nstokes"])
        # map_solution = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(map_solution), covariance_unity, lmin=0, n_iter=n_iter)
        covariance_unity = np.zeros((param_dict['lmax']+1,param_dict["nstokes"],param_dict["nstokes"]))
        covariance_unity[lmin:,...] = np.eye(param_dict["nstokes"])
        first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt(np.copy(first_member), covariance_unity, lmin=0, n_iter=n_iter)

    second_member = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_y, np.linalg.pinv(red_cov_approx_matrix_sqrt), lmin=lmin, n_iter=n_iter)
    
    map_solution_0 = first_member + second_member
    map_solution = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_solution_0.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix_sqrt, lmin=lmin, n_iter=n_iter)

    return map_solution


def get_fluctuating_term_maps(param_dict, red_cov_matrix, BtinvNB, BtinvN_sqrt, map_random_realization_xi=[], map_random_realization_chi=[], initial_guess=[], lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
    """ 
        Solve fluctuation term with formulation (C^-1 + N^-1) for the left member :
        (C^{-1} + E^t (B^t N^{-1} B)^{-1} E) \zeta = C^{-1/2} xi + (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} chi

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

    red_inverse_cov_matrix = np.linalg.pinv(red_cov_matrix)
    

    # Creation of the random maps
    if len(map_random_realization_xi) == 0:
        print("Recalculating xi !")
        map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_random_realization_chi) == 0:
        print("Recalculating chi !")
        map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))

    # Computation of the right side member of the CG
    red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(red_inverse_cov_matrix)

    # First right member : C^{-1/2} \xi
    right_member_1 = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_realization_xi, red_inv_cov_sqrt, lmin=lmin, n_iter=n_iter)

    ## Left hand side term : (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi
    right_member_2 = np.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, map_random_realization_chi)[0]/BtinvNB[0,0] # Selecting CMB component of the random variable

    right_member = (right_member_1 + right_member_2).ravel()
    
    # Computation of the left side member of the CG
    
    # First left member : C^{-1} 
    first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_inverse_cov_matrix, lmin=lmin, n_iter=n_iter)
    
    ## Second left member : (E^t (B^t N^{-1} B) E)
    def second_term_left(x, number_component=param_dict['number_components']):
        # cg_variable = x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))
        # x_all_components = np.zeros((number_component, cg_variable.shape[0], cg_variable.shape[1]))
        # x_all_components[0,...] =cg_variable
        # return np.einsum('kc,csp->ksp', np.linalg.pinv(BtinvNB), x_all_components)[0]
        return x/BtinvNB[0,0]

    func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()
    # Initial guess for the CG
    if len(initial_guess) == 0:
        initial_guess = np.zeros_like(map_random_realization_xi)

    # Actual start of the CG
    fluctuating_map, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    print("CG-Python-0 Fluct finished in ", number_iterations, "iterations !!")    

    if exit_code != 0:
        print("CG didn't converge with fluctuating term ! Exitcode :", exit_code, flush=True)
    return fluctuating_map.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))


def solve_generalized_wiener_filter_term(param_dict, s_cML, red_cov_matrix, BtinvNB, initial_guess=[], lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
    """ 
        Solve Wiener filter term with formulation (1 + C^1/2 N^-1 C^1/2) for the left member

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax

        s_cML : data maps, for Wiener filter CG ; dimensions [nstokes, npix]

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
    if param_dict['nstokes'] != 1:
        assert s_cML.shape[0] == param_dict['nstokes']
        assert s_cML.shape[1] == 12*param_dict['nside']**2
    

    # Computation of the right side member of the CG
    s_cML_extended = np.zeros((param_dict['number_components'], s_cML.shape[0], s_cML.shape[1]))
    s_cML_extended[0,...] = s_cML
    # right_member = np.einsum('kc,csp->ksp', np.linalg.pinv(BtinvNB), s_cML_extended)[0].ravel() # Selecting CMB component of the
    right_member = (s_cML/BtinvNB[0,0]).ravel()

    # Computation of the left side member of the CG
    first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.linalg.pinv(red_cov_matrix), lmin=lmin, n_iter=n_iter).ravel()
    
    ## Second left member : (E^t (B^t N^{-1} B)^{-1}
    def second_term_left(x, number_component=param_dict['number_components']):
        # cg_variable = x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))
        # x_all_components = np.zeros((number_component, cg_variable.shape[0], cg_variable.shape[1]))
        # x_all_components[0,...] = cg_variable
        # return np.einsum('kc,csp->ksp', np.linalg.pinv(BtinvNB), x_all_components)[0]
        return x/BtinvNB[0,0]

    func_left_term = lambda x : first_term_left(x).ravel() + second_term_left(x).ravel()

    # Initial guess for the CG
    if len(initial_guess) == 0:
        initial_guess = np.zeros_like(s_cML)

    # Actual start of the CG
    wiener_filter_term, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")

    if exit_code != 0:
        print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
    return wiener_filter_term.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))



def get_inverse_wishart_sampling_from_c_ells(sigma_ell, q_prior=0, l_min=0, option_ell_2=2):
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

        # # Option sampling with brute force inverse Wishart
        # if option_ell_2 == -1:
        #     print("~~Not caring about inverse Wishart distribution not defined !", flush=True)
        #     mean = np.zeros(nstokes)
        #     sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*q_prior))
        #     sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)

        # Option sampling with Jeffrey prior
        if option_ell_2 == 0:
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


def get_inverse_gamma_sampling_only_BB_from_c_ells(sigma_ell, q_prior=0, l_min=0, option_ell_2=2):
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
    # sampling_Wishart = np.zeros_like(invert_parameter_Wishart)
    


    # for ell in range(max(lmin,2),lmax+1):
    #     sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell], size=(2*ell - nstokes + 2*q_prior))
    #     sampling_Wishart[ell] = np.dot(sample_gaussian.T,sample_gaussian)
    # # sampling_Wishart[max(lmin,2):,...] = np.einsum('lkj,lkm->ljm',sample_gaussian,sample_gaussian)
    # return np.linalg.pinv(sampling_Wishart)
    sampling_Gamma = np.zeros_like(invert_parameter_Wishart)
    all_betas = invert_parameter_Wishart[lmin:,1,1]/2
    all_alphas = (2*np.arange(lmin,lmax+1) + 1 - 2*nstokes)/2
    sampling_Gamma[lmin:,1,1] = scipy.stats.invgamma.rvs(all_alphas, size=(lmax+1-lmin))*all_betas
    return sampling_Gamma
    

def get_inverse_operators_harm_pixel(param_dict, right_member, operator_harmonic, operator_pixel, initial_guess=[], lmin=2, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), with_prints=False):
    """ Solve the CG given by :
        (operator_harmonic + operator_pixel) variable = right_member

        with operator_harmonic an operator acting on harmonic domain
        with operator_pixel an operator acting on (component,pixel_domain) domain, in CMB component

        Returns
        -------
        A pixel map
        
    """

    first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), operator_harmonic, lmin=lmin, n_iter=n_iter)
    
    ## Second left member : (E^t (B^t N^{-1} B)^{-1}
    def second_term_left(x, number_component=param_dict['number_components']):
        cg_variable = x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))
        x_all_components = np.zeros((number_component, cg_variable.shape[0], cg_variable.shape[1]))
        x_all_components[0,...] = cg_variable
        return np.einsum('kc,csp->ksp', operator_pixel, x_all_components)[0]

    func_left_term = lambda x : (first_term_left(x) + second_term_left(x)).ravel()

    if len(initial_guess) == 0:
        initial_guess = np.zeros((param_dict["nstokes"],12*param_dict["nside"]**2))
    inverse_term, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member.ravel(), limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    if with_prints:
        print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")
        if exit_code != 0:
            print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
    return inverse_term.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))

def get_conditional_proba_mixing_matrix_foregrounds(params_mixing_matrix, mixingmatrix_object, full_data_without_CMB, eta_prime_maps, freq_inverse_noise, red_cov_approx_matrix, param_dict, lmin, n_iter, limit_iter_cg, tolerance, with_prints=False, regularization_constant=10**10):
    """ Get conditional probability of mixing matrix by sampling it using emcee

        The associated conditional probability is given by : 
        - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        + \eta^t N_c^{1/2} (C_{approx} + E^t (B^T N^{-1} B)^{-1} E)^{-1} N_c^{1/2} \ \eta 
    """

    # print("Test params :", params_mixing_matrix.reshape((param_dict['number_frequencies']-2,param_dict['number_components']-1)))
    mixingmatrix_object.update_params(params_mixing_matrix.reshape((param_dict['number_frequencies']-2,param_dict['number_components']-1)))

    # Building the first term : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
    complete_mixing_matrix_fg = mixingmatrix_object.get_B_fgs()

    BtinvNB_fg = get_inv_BtinvNB(freq_inverse_noise, complete_mixing_matrix_fg)
    cp_freq_inv_noise_fg = get_BtinvN(freq_inverse_noise, complete_mixing_matrix_fg)

    full_data_without_CMB_with_noise = np.einsum('cf,fsp->csp', cp_freq_inv_noise_fg, full_data_without_CMB)
    # print("Test 1 :", np.mean(full_data_without_CMB_with_noise), np.max(full_data_without_CMB_with_noise), np.min(full_data_without_CMB_with_noise), full_data_without_CMB_with_noise)
    first_term_complete = np.einsum('psc,cm,msp', full_data_without_CMB_with_noise.T, BtinvNB_fg, full_data_without_CMB_with_noise)
    # print("Test 2 :", np.mean(BtinvNB_fg), np.max(BtinvNB_fg), np.min(BtinvNB_fg), BtinvNB_fg)

    # Building the second term term \eta^t N_c^{1/2] (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
    complete_mixing_matrix = mixingmatrix_object.get_B()
    BtinvNB = get_inv_BtinvNB(freq_inverse_noise, complete_mixing_matrix)
    BtinvN_sqrt = get_BtinvN(scipy.linalg.sqrtm(freq_inverse_noise), complete_mixing_matrix)

    ## Left hand side term : N_c^{1/2] \eta = (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \eta
    # noise_weighted_eta = np.einsum('kc,cf,fsp->ksp', BtinvNB, BtinvN_sqrt, eta_maps)[0] # Selecting CMB component
    eta_prime_maps_extended = np.zeros((param_dict['number_components'],param_dict['nstokes'],12*param_dict['nside']**2))
    eta_prime_maps_extended[0] = eta_prime_maps
    noise_weighted_eta = np.einsum('kc,csp->ksp', BtinvNB, eta_prime_maps_extended)[0] # Selecting CMB component

    # Then getting (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
    operator_harmonic = red_cov_approx_matrix
    operator_pixel = BtinvNB
    inverse_term = get_inverse_operators_harm_pixel(param_dict, noise_weighted_eta, operator_harmonic, operator_pixel, initial_guess=[], lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance, with_prints=with_prints)

    # And finally \eta^t N_c^{1/2] (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
    # second_term_complete = np.einsum('fsk,fsk', noise_weighted_eta, inverse_term)
    second_term_complete = np.einsum('sk,sk', noise_weighted_eta, inverse_term)
    # print("Test", first_term_complete, second_term_complete)
    return -(-first_term_complete + second_term_complete)/2. + regularization_constant



def sample_mixing_matrix_term(param_dict, mixingmatrix_object, full_data_without_CMB, eta_prime_maps, red_cov_approx_matrix, freq_inverse_noise, lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), n_walkers=1, number_steps_sampler=1000, with_prints=False):
    """ Solve sampling step 4 : sampling B_f
        Sample mixing matrix with formualtion : -(d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c) + eta^t (S_{approx} + E^t (B^T N^{-1} B)^{-1} E) eta

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax

        full_data : full data maps, for Wiener filter CG ; dimensions [nstokes, npix]

        red_cov_matrix : covariance matrices in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]

        
        lmin : minimum multipole to be considered, default 0
        
        n_iter : number of iterations for harmonic computations, default 8

        Returns
        -------
        Mixing matrix sampled
    """

    # assert initial_guess_fg_mixing_matrix.shape[0] == param_dict['number_components']-1
    # assert initial_guess_fg_mixing_matrix.shape[1] == param_dict['number_frequencies']

    initial_guess_fg_mixing_matrix = mixingmatrix_object.params.ravel()
    dimensions_mixing_matrix = len(initial_guess_fg_mixing_matrix)
    
    # sample_params_mixing_matrix_FG = emcee.EnsembleSampler(n_walkers, dimensions_mixing_matrix, get_conditional_proba_mixing_matrix_foregrounds, args=[mixingmatrix_object, full_data_without_CMB, eta_maps, freq_inverse_noise, red_cov_approx_matrix, param_dict, lmin, n_iter, limit_iter_cg, tolerance, with_prints])
    sample_params_mixing_matrix_FG = emcee.EnsembleSampler(n_walkers, dimensions_mixing_matrix, get_conditional_proba_mixing_matrix_foregrounds, args=[mixingmatrix_object, full_data_without_CMB, eta_prime_maps, freq_inverse_noise, red_cov_approx_matrix, param_dict, lmin, n_iter, limit_iter_cg, tolerance, with_prints])

    # full_initial_guess = np.tile(initial_guess_fg_mixing_matrix, n_walkers).reshape((dimensions_mixing_matrix,n_walkers)).T
    # full_initial_guess = ((np.random.random(size=n_walkers*dimensions_mixing_matrix) - .5)*20).reshape((n_walkers,dimensions_mixing_matrix))
    
    # full_initial_guess = np.random.randn(n_walkers,dimensions_mixing_matrix)


    full_initial_guess = np.random.uniform(low=initial_guess_fg_mixing_matrix*.25,high=initial_guess_fg_mixing_matrix*.75, size=(n_walkers,dimensions_mixing_matrix))
    print("Test", full_initial_guess)
    final_position_walkers = sample_params_mixing_matrix_FG.run_mcmc(full_initial_guess, number_steps_sampler)

    # return sample_params_mixing_matrix_FG.get_last_sample()
    return sample_params_mixing_matrix_FG.get_chain()

