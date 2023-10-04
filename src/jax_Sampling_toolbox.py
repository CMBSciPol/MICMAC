import os, sys, time
import numpy as np
import healpy as hp
from collections import namedtuple
from jax import random, dtypes
import jax.numpy as jnp
import jax.scipy as jsp
import jax_healpy as jhp
from functools import partial
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC

from .jax_tools import *
from .noisecovar import *


MHState = namedtuple("MHState", ["u", "rng_key"])

class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "u"

    def __init__(self, potential_fn, step_size=0.1):
        self.potential_fn = potential_fn
        self.step_size = step_size

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        accept_prob = jnp.exp(self.potential_fn(u, **model_kwargs) - self.potential_fn(u_proposal, **model_kwargs))
        u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
        return MHState(u_new, rng_key)




def get_sampling_eta_prime_JAX(param_dict, red_cov_approx_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_x=jnp.empty(0), map_random_y=jnp.empty(0), jax_key_PNRG=jax.random.PRNGKey(1), lmin=0, n_iter=8):
    """ Solve sampling step 1 : sampling eta'
        Solve CG for eta term with formulation : eta' = C_approx^(1/2) x + (B^t N^{-1} B)^{-1} B^T N^{-1/2} y

        Parameters
        ----------
        param_dict : dictionnary containing the following fields : nside, nstokes, lmax, number_frequencies
        
        red_cov_approx_matrix : correction covariance matrice (C_approx) in harmonic domain, dimension [lmin:lmax, nstokes, nstokes]
        cp_cp_noise : matrices of noise combined with mixing matrices corresponding to (B^t N^{-1} B)^{-1}, dimension [component, component]
        # cp_freq_inv_noise_sqrt : matrices of noise combined with mixing matrices corresponding to B^T N^{-1/2}, dimension [component, frequencies]

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
    if jnp.size(map_random_x) == 0:
        print("Recalculating x !")
        # map_random_x = np.random.normal(loc=0, scale=1/jhp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
        map_random_x = jax.random.normal(jax_key_PNRG, shape=(param_dict["nstokes"],12*param_dict["nside"]**2))/jhp.nside2resol(param_dict["nside"])
    if jnp.size(map_random_y) == 0:
        print("Recalculating y !")
        # map_random_y = np.random.normal(loc=0, scale=1/jhp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))
        map_random_y = jax.random.normal(jax_key_PNRG+1, shape=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))/jhp.nside2resol(param_dict["nside"])
    # Computation of the right hand side member of the CG
    red_cov_approx_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_cov_approx_matrix)

    # First right member : C_approx^{-1/2} x
    # first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt(map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix_sqrt, lmin=lmin, n_iter=n_iter)
    first_member = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_random_x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_approx_matrix_sqrt, nside=param_dict["nside"], lmin=lmin, n_iter=n_iter)
    # # Second right member : E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2}
    second_member = jnp.einsum('kc,cf,fsp->ksp', cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_y)[0] # Selecting CMB component of the random variable

    return first_member + second_member


def get_fluctuating_term_maps_JAX(param_dict, red_cov_matrix, cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_xi=jnp.empty(0), map_random_realization_chi=jnp.empty(0), jax_key_PNRG=jax.random.PRNGKey(10), initial_guess=jnp.empty(0), lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
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

    # assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin

    red_inverse_cov_matrix = jnp.linalg.pinv(red_cov_matrix)
    

    # Creation of the random maps
    if jnp.size(map_random_realization_xi) == 0:
        print("Recalculating xi !")
        # map_random_realization_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
        map_random_realization_xi = jax.random.normal(jax_key_PNRG, shape=(param_dict["nstokes"],12*param_dict["nside"]**2))/jhp.nside2resol(param_dict["nside"])
    if jnp.size(map_random_realization_chi) == 0:
        print("Recalculating chi !")
        # map_random_realization_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))
        map_random_realization_chi = jax.random.normal(jax_key_PNRG+1, shape=(param_dict["number_frequencies"],param_dict["nstokes"],12*param_dict["nside"]**2))/jhp.nside2resol(param_dict["nside"])

    # Computation of the right side member of the CG
    red_inv_cov_sqrt = get_sqrt_reduced_matrix_from_matrix_jax(red_inverse_cov_matrix)

    # First right member : C^{-1/2} \xi
    right_member_1 = maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(map_random_realization_xi, red_inv_cov_sqrt, nside=param_dict["nside"], lmin=lmin, n_iter=n_iter)

    ## Left hand side term : (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \chi
    right_member_2 = jnp.einsum('kc,cf,fsp->ksp', cp_cp_noise, cp_freq_inv_noise_sqrt, map_random_realization_chi)[0] # Selecting CMB component of the random variable

    right_member = (right_member_1 + right_member_2).ravel()

    # Computation of the left side member of the CG
    
    # First left member : C^{-1} 
    first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_inverse_cov_matrix, nside=param_dict["nside"], lmin=lmin, n_iter=n_iter)
    
    ## Second left member : (E^t (B^t N^{-1} B) E)
    def second_term_left(x, number_component=param_dict['number_components']):
        cg_variable = x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))
        x_all_components = jnp.zeros((number_component, cg_variable.shape[0], cg_variable.shape[1]))
        x_all_components[0,...] = cg_variable
        return jnp.einsum('kc,csp->ksp', jnp.linalg.pinv(cp_cp_noise), x_all_components)[0]

    func_left_term = lambda x : (first_term_left(x) + second_term_left(x)).ravel()
    # Initial guess for the CG
    if jnp.size(initial_guess) == 0:
        initial_guess = jnp.zeros_like(map_random_realization_xi)

    # Actual start of the CG
    # fluctuating_map, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    fluctuating_map, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=tolerance, atol=tolerance, maxiter=limit_iter_cg)
    print("CG-Python-0 Fluct finished in ", number_iterations, "iterations !!")    

    # if exit_code != 0:
    #     print("CG didn't converge with fluctuating term ! Exitcode :", exit_code, flush=True)
    return fluctuating_map.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))


def solve_generalized_wiener_filter_term_JAX(param_dict, s_cML, red_cov_matrix, cp_cp_noise, initial_guess=[], lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12)):
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

    # assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    # if param_dict['nstokes'] != 1:
    #     assert s_cML.shape[0] == param_dict['nstokes']
    #     assert s_cML.shape[1] == 12*param_dict['nside']**2
    

    # Computation of the right side member of the CG
    s_cML_extended = jnp.zeros((param_dict['number_components'], s_cML.shape[0], s_cML.shape[1]))
    s_cML_extended[0,...] = s_cML
    right_member = jnp.einsum('kc,csp->ksp', jnp.linalg.pinv(cp_cp_noise), s_cML_extended)[0].ravel() # Selecting CMB component of the

    # Computation of the left side member of the CG
    first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), jnp.linalg.pinv(red_cov_matrix), lmin=lmin, n_iter=n_iter)
    
    ## Second left member : (E^t (B^t N^{-1} B)^{-1}
    def second_term_left(x, number_component=param_dict['number_components']):
        cg_variable = x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))
        x_all_components = jnp.zeros((number_component, cg_variable.shape[0], cg_variable.shape[1]))
        x_all_components[0,...] = cg_variable
        return jnp.einsum('kc,csp->ksp', jnp.linalg.pinv(cp_cp_noise), x_all_components)[0]

    func_left_term = lambda x : (first_term_left(x) + second_term_left(x)).ravel()

    # Initial guess for the CG
    if len(initial_guess) == 0:
        initial_guess = jnp.zeros_like(s_cML)

    # Actual start of the CG
    # wiener_filter_term, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    wiener_filter_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=tolerance, atol=tolerance, maxiter=limit_iter_cg)
    print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")

    # if exit_code != 0:
    #     print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
    return wiener_filter_term.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))



def get_inverse_wishart_sampling_from_c_ells_JAX(sigma_ell_argument, jax_key_PNRG=jax.random.PRNGKey(100), q_prior=0, l_min=0, option_ell_2=2):
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

    sigma_ell = jnp.copy(sigma_ell_argument)

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
        sigma_ell[i] *= 2*jnp.arange(lmax+1) + 1

    lmin = l_min

    invert_parameter_Wishart = jnp.linalg.pinv(get_reduced_matrix_from_c_ell_jax(sigma_ell))

    # assert invert_parameter_Wishart.shape[0] == lmax + 1 #- lmin
    sampling_Wishart = jnp.zeros_like(invert_parameter_Wishart)

    # assert (option_ell_2 == 0) or (option_ell_2 == 1) or (option_ell_2 == 2)
    # Option sampling without caring about inverse Wishart not defined
    ell_2 = 2
    if l_min <= 2 and (2*ell_2 + 1 - 2*nstokes + 2*q_prior <= 0):
        # 2*ell_2 + 1 - 2*nstokes + 2*q_prior <= 0) correspond to the definition condition of the inverse Wishart distribution

        # Option sampling with Jeffrey prior
        if option_ell_2 == 0:
            Jeffrey_prior = 1
            print("~~Applying Jeffry prior for ell=2 !", flush=True)
            mean = jnp.zeros(nstokes)
            # sample_gaussian = np.random.multivariate_normal(mean, invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*Jeffrey_prior))
            sample_gaussian = jax.random.multivariate_normal(jax_key_PNRG, mean, invert_parameter_Wishart[ell_2], shape=[2*ell_2 - nstokes + 2*Jeffrey_prior])
            sampling_Wishart[ell_2] = jnp.einsum('ij,ik->jk', sample_gaussian,sample_gaussian)

        # Option sampling separately TE and B
        elif option_ell_2 == 1:
            print("~~Sampling separately TE and B for ell=2 !", flush=True)
            invert_parameter_Wishart_2 = jnp.zeros((nstokes,nstokes))
            reduced_matrix_2 = get_reduced_matrix_from_c_ell_jax(sigma_ell)[ell_2]
            invert_parameter_Wishart_2[:nstokes-1, :nstokes-1] = jnp.linalg.pinv(reduced_matrix_2[:nstokes-1,:nstokes-1])
            invert_parameter_Wishart_2[nstokes-1, nstokes-1] = 1/reduced_matrix_2[nstokes-1,nstokes-1]
            # sample_gaussian_TE = np.random.multivariate_normal(np.zeros(nstokes-1), invert_parameter_Wishart_2[:nstokes-1, :nstokes-1], size=(2*ell_2 - (nstokes-1)))
            # sample_gaussian_B = np.random.normal(loc=0, scale=invert_parameter_Wishart_2[nstokes-1, nstokes-1], size=(2*ell_2 - 1))
            sample_gaussian_TE = jax.random.multivariate_normal(jax_key_PNRG+1, jnp.zeros(nstokes-1), invert_parameter_Wishart_2[:nstokes-1, :nstokes-1], shape=[2*ell_2 - (nstokes-1)])
            sample_gaussian_B = jax.random.normal(jax_key_PNRG+2, shape=(2*ell_2 - 1))*invert_parameter_Wishart_2[nstokes-1, nstokes-1]
            sampling_Wishart[ell_2][:nstokes-1,:nstokes-1] = jnp.einsum('ij,ik->jk', sample_gaussian_TE,sample_gaussian_TE)
            sampling_Wishart[ell_2][nstokes-1,nstokes-1] = jnp.einsum('i,i', sample_gaussian_B.T,sample_gaussian_B)

        lmin = 3

    for ell in range(max(lmin,2),lmax+1):
        # sample_gaussian = np.random.multivariate_normal(jnp.zeros(nstokes), invert_parameter_Wishart[ell], size=(2*ell - nstokes + 2*q_prior))
        sample_gaussian = jax.random.multivariate_normal(jax_key_PNRG+3+ell, jnp.zeros(nstokes), invert_parameter_Wishart[ell], shape=[2*ell  - nstokes + 2*q_prior])
        sampling_Wishart[ell] = jnp.einsum('ij,ik->jk', sample_gaussian,sample_gaussian)
    # sampling_Wishart[max(lmin,2):,...] = np.einsum('lkj,lkm->ljm',sample_gaussian,sample_gaussian)
    return jnp.linalg.pinv(sampling_Wishart)


# @partial(jax.jit, static_argnames=['number_components', 'nstokes', 'nside', 'right_member', 'operator_harmonic', 'operator_pixel', 'lmin', 'n_iter', 'limit_iter_cg', 'tolerance', 'with_prints'])
def get_inverse_operators_harm_pixel_JAX(number_components, nstokes, nside, right_member, operator_harmonic, operator_pixel, initial_guess=[], lmin=2, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), with_prints=False):
    """ Solve the CG given by :
        (operator_harmonic + operator_pixel) variable = right_member

        with operator_harmonic an operator acting on harmonic domain
        with operator_pixel an operator acting on (component,pixel_domain) domain, in CMB component

        Returns
        -------
        A pixel map
        
    """

    # print("Test 2 :", lmin, flush=True)
    # first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), operator_harmonic, lmin=lmin, n_iter=n_iter)
    first_term_left = lambda x : maps_x_reduced_matrix_generalized_sqrt_sqrt_JAX_compatible(x.reshape((nstokes,12*nside**2)), operator_harmonic, nside=nside, lmin=lmin, n_iter=n_iter)
    ## Second left member : (E^t (B^t N^{-1} B)^{-1}
    # @partial(jax.jit, static_argnames=['operator_pixel','param_dict'])
    def second_term_left(x, operator_pixel=operator_pixel):
        cg_variable = x.reshape((nstokes,12*nside**2))
        x_all_components = jnp.zeros((number_components, cg_variable.shape[0], cg_variable.shape[1]))
        x_all_components = x_all_components.at[0].set(cg_variable)
        return jnp.einsum('kc,csp->ksp', operator_pixel, x_all_components)[0]

    func_left_term = lambda x : (first_term_left(x) + second_term_left(x)).ravel()

    if len(initial_guess) == 0:
        initial_guess = jnp.zeros((nstokes,12*nside**2))
    # inverse_term, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member.ravel(), limit_iter_cg=limit_iter_cg, tolerance=tolerance)
    inverse_term, number_iterations = jsp.sparse.linalg.cg(func_left_term, right_member.ravel(), x0=initial_guess.ravel(), tol=tolerance, atol=tolerance, maxiter=limit_iter_cg)
    if with_prints:
        print("CG-Python-0 WF finished in ", number_iterations, "iterations !!")
        # if exit_code != 0:
        #     print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
    return inverse_term.reshape((nstokes, 12*nside**2))


@partial(jax.jit, static_argnames=['number_components', 'nstokes', 'nside', 'lmin', 'n_iter', 'limit_iter_cg', 'tolerance', 'with_prints', 'regularization_constant', 'regularization_factor'])
def get_conditional_proba_mixing_matrix_foregrounds_alternative_JAX(complete_mixing_matrix, full_data_without_CMB, eta_prime_maps, freq_inverse_noise, red_cov_approx_matrix, number_components, nstokes, nside, lmin, n_iter, limit_iter_cg, tolerance, with_prints=False, regularization_constant=-1, regularization_factor=10**10):
    """ Get conditional probability of mixing matrix by sampling it using emcee

        The associated conditional probability is given by : 
        - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
        + \eta^t N_c^{1/2} (C_{approx} + E^t (B^T N^{-1} B)^{-1} E)^{-1} N_c^{1/2} \ \eta 
    """

    # print("Test params :", params_mixing_matrix.reshape((param_dict['number_frequencies']-2,param_dict['number_components']-1)))
    # mixingmatrix_object = mixing_matrix_obj
    # mixingmatrix_object.update_params(params_mixing_matrix.reshape((param_dict['number_frequencies']-2,param_dict['number_components']-1)))
    # mixingmatrix_object = katame.MixingMatrix(frequency_list, param_dict['number_components'], params_mixing_matrix, pos_special_freqs=pos_special_freqs)

    # Building the first term : - (d - B_c s_c)^t N^{-1} B_f (B_f^t N^{-1} B_f)^{-1} B_f^t N^{-1} (d - B_c s_c)
    # complete_mixing_matrix_fg = mixingmatrix_object.get_B_fgs()
    complete_mixing_matrix_fg = complete_mixing_matrix[:,1:]

    cp_cp_noise_fg = get_inv_BtinvNB(freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)
    cp_freq_inv_noise_fg = get_BtinvN(freq_inverse_noise, complete_mixing_matrix_fg, jax_use=True)

    full_data_without_CMB_with_noise = jnp.einsum('cf,fsp->csp', cp_freq_inv_noise_fg, full_data_without_CMB)
    # print("Test 1 :", np.mean(full_data_without_CMB_with_noise), np.max(full_data_without_CMB_with_noise), np.min(full_data_without_CMB_with_noise), full_data_without_CMB_with_noise)
    first_term_complete = jnp.einsum('psc,cm,msp', full_data_without_CMB_with_noise.T, cp_cp_noise_fg, full_data_without_CMB_with_noise)
    # print("Test 2 :", np.mean(cp_cp_noise_fg), np.max(cp_cp_noise_fg), np.min(cp_cp_noise_fg), cp_cp_noise_fg)

    # Building the second term term \eta^t N_c^{1/2] (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
    # complete_mixing_matrix = mixingmatrix_object.get_B()
    cp_cp_noise = get_inv_BtinvNB(freq_inverse_noise, complete_mixing_matrix, jax_use=True)

    ## Left hand side term : N_c^{1/2] \eta = (E^t (B^t N^{-1} B)^{-1} B^t N^{-1/2} \eta
    # noise_weighted_eta = np.einsum('kc,cf,fsp->ksp', cp_cp_noise, cp_freq_inv_noise_sqrt, eta_maps)[0] # Selecting CMB component
    # eta_prime_maps_extended = jnp.array(np.zeros((param_dict['number_components'],param_dict['nstokes'],12*param_dict['nside']**2)))
    eta_prime_maps_extended = jnp.zeros((number_components,nstokes,12*nside**2))
    # eta_prime_maps_extended[0] = eta_prime_maps
    eta_prime_maps_extended = eta_prime_maps_extended.at[0].set(eta_prime_maps)
    noise_weighted_eta = jnp.einsum('kc,csp->ksp', cp_cp_noise, eta_prime_maps_extended)[0] # Selecting CMB component

    # Then getting (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
    operator_harmonic = red_cov_approx_matrix
    operator_pixel = cp_cp_noise
    # print("Test 3 :", lmin, flush=True)
    inverse_term = get_inverse_operators_harm_pixel_JAX(number_components, nstokes, nside, noise_weighted_eta, operator_harmonic, operator_pixel, initial_guess=[], lmin=lmin, n_iter=n_iter, limit_iter_cg=limit_iter_cg, tolerance=tolerance, with_prints=with_prints)

    # And finally \eta^t N_c^{1/2] (C_approx + E^t (B^t N^{-1} B)^{-1} E)^{-1} N_c^{1/2] \eta
    # second_term_complete = np.einsum('fsk,fsk', noise_weighted_eta, inverse_term)
    second_term_complete = jnp.einsum('sk,sk', noise_weighted_eta, inverse_term)
    # print("Test", first_term_complete, second_term_complete)
    return -(-first_term_complete + second_term_complete)/2./regularization_factor + regularization_constant
