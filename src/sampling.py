import os, sys, time
import numpy as np
import healpy as hp
import scipy
# import mappraiser_wrapper_WF as mappraiser
from .tools import *
from .proba_functions import *

def get_inverse_gamma_distribution(pixel_maps, lmax=None, n_iter=0):
    """ Assume pixel_maps containt T, Q and U

        Build Gaussian variate with 0 mean and unit variance, of the form of a (2 ell - 1) vector
    """

    nstokes = len(pixel_maps.shape)
    if nstokes > 1:
        nstokes = pixel_maps.shape[0]

    c_ells_observed = hp.anafast(pixel_maps, lmax=lmax, iter=n_iter) # Add 2 ell +1
    for i in range(nstokes):
        c_ells_observed[i] *= 2*np.arange(lmax) + 1

    gaussian_variate = np.zeros((nstokes, lmax))
    ell_values_cond = ((np.arange(lmax*(2*lmax-1))%(2*lmax-1)).reshape((2*lmax-1, lmax), order='F') < 2*np.arange(lmax)-1).T

    ell_values_cond_all_nstokes = np.stack((ell_values_cond,ell_values_cond,ell_values_cond))
    
    gaussian_variates = np.random.normal(loc=0, scale=1, size=(nstokes, lmax, 2*lmax-1))
    
    gaussian_variates[np.logical_not(ell_values_cond_all_nstokes)] = 0
    gaussian_variate = np.sum(np.power(gaussian_variates, 2), axis=2)

    return c_ells_observed[:nstokes,1:]/gaussian_variate[:,1:]


def get_inverse_wishart_sampling_from_c_ells(sigma_ell, lmax=None, q_prior=0, l_min=0, option_ell_2=1):
    """ Assume sigma_ell to be with all ells
        
        Also assumes the monopole and dipole to be 0

        q_prior = 0 : uniform prior
        q_prior = 1 : Jeffrey prior

    """

    if len(sigma_ell.shape) == 1:
        nstokes == 1
    elif sigma_ell.shape[0] == 6:
        nstokes = 3
    elif sigma_ell.shape[0] == 3:
        nstokes = 2

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
            print("Not caring about inverse Wishart distribution not defined !", flush=True)
            mean = np.zeros(nstokes)
            sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*q_prior))
            sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)
            
        # Option sampling with Jeffrey prior
        elif option_ell_2 == 0:
            Jeffrey_prior = 1
            print("Applying Jeffry prior for ell=2 !", flush=True)
            mean = np.zeros(nstokes)
            sample_gaussian = np.random.multivariate_normal(mean, invert_parameter_Wishart[ell_2], size=(2*ell_2 - nstokes + 2*Jeffrey_prior))
            sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)

        # Option sampling separately TE and B
        elif option_ell_2 == 1:
            print("Sampling separately TE and B for ell=2 !", flush=True)
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


def get_fluctuating_term_maps(param_dict, initial_guess, red_inverse_cov_matrix, red_inverse_noise, comm_world, map_white_noise_xi, map_white_noise_chi, language='Python', lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), iter_alm=1000, tolerance_alm=10**(-10)):
    """
        Solve Wiener filter term with formulation (C^-1 + N^-1) for the left member, 
        and with the use of matrix square root to apply in harmonic space to the alms
    """
    assert (language == 'C') or (language == 'Python')
    assert red_inverse_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin

    red_cov_sqrt_inv = get_sqrt_reduced_matrix_from_matrix(red_inverse_cov_matrix)
    # red_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(np.linalg.pinv(red_inverse_cov_matrix), lmin=lmin)
    red_inv_noise_sqrt = get_sqrt_reduced_matrix_from_matrix(red_inverse_noise)

    # new_matrix_cov = np.zeros_like(red_inverse_noise)
    # lmin_to_use = np.where(lmin <= 2, 2, 0)
    # print("Shapes : red_inverse_cov_matrix", red_inverse_cov_matrix.shape, "red_inverse_noise", red_inverse_noise.shape, flush=True)
    # new_matrix_cov[lmin_to_use:,...] = red_inverse_cov_matrix[lmin_to_use:,...] + red_inverse_noise[lmin_to_use:,...]
    new_matrix_cov = red_inverse_cov_matrix + red_inverse_noise
    # print("Test 0", new_matrix_cov[:20,...])
    # red_eye = np.zeros_like(red_inverse_noise)
    # red_eye[2:,...] = np.repeat(np.eye(param_dict['nstokes']),new_matrix_cov.shape[0]-2).reshape((new_matrix_cov.shape[0]-2,param_dict['nstokes'],param_dict['nstokes']),order='F')

    if len(map_white_noise_xi) == 0:
        map_white_noise_xi = np.random.normal(loc=0, scale=1, size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    if len(map_white_noise_chi) == 0:
        map_white_noise_chi = np.random.normal(loc=0, scale=1, size=(param_dict["nstokes"],12*param_dict["nside"]**2))

    if language == 'Python':
        right_member_1 = maps_x_reduced_matrix_generalized_sqrt(map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_sqrt_inv, lmin=lmin, n_iter=n_iter)

        right_member_2 = maps_x_reduced_matrix_generalized_sqrt(map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_inv_noise_sqrt, lmin=lmin, n_iter=n_iter)

        right_member = (right_member_1 + right_member_2).ravel()

        # func_left_term = lambda x : (maps_x_reduced_matrix_generalized_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_matrix_cov, lmin=lmin, n_iter=n_iter)).ravel()
        new_matrix_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(new_matrix_cov)
        func_left_term = lambda x : (maps_x_reduced_matrix_generalized_sqrt_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_matrix_cov_sqrt, lmin=lmin, n_iter=n_iter)).ravel()

    elif language == 'C':
        raise Exception("C not added in this version")
        # CMB_map_Wiener_filter_0 = np.zeros(((param_dict["nstokes"],12*param_dict["nside"]**2)))


        # new_matrix_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(new_matrix_cov)

        # red_sqrt_covariance_matrix_sqrt_inv = get_sqrt_reduced_matrix_from_matrix(red_cov_sqrt_inv)

        # red_inv_noise_sqrt_sqrt = get_sqrt_reduced_matrix_from_matrix(red_inv_noise_sqrt)

        # right_member_1 = mappraiser.apply_red_matrix_x_alm(param_dict, map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), red_sqrt_covariance_matrix_sqrt_inv, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)
        # right_member_2 = mappraiser.apply_red_matrix_x_alm(param_dict, map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), red_inv_noise_sqrt_sqrt, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)
        # right_member = (right_member_1 + right_member_2).ravel()
        # func_left_term = lambda x : (mappraiser.apply_red_matrix_x_alm(param_dict, x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), new_matrix_cov_sqrt, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)).ravel()

    fluctuating_map, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)

    if language == 'Python':
        print("CG-Python-0 Fluct sqrt finished in ", number_iterations, "iterations for fluctuating term !!")    
        # fluctuating_map = pss.maps_x_reduced_matrix_generalized_sqrt(fluctuating_map_y.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_sqrt, lmin=lmin, n_iter=n_iter)
    
    # elif language == 'C':
    #     print("CG-C-0 Fluct w/ iter finished in ", number_iterations, "iterations for fluctuating term !!")
        
        # fluctuating_map = mappraiser.apply_red_matrix_x_alm(param_dict, fluctuating_map_y.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), red_sqrt_covariance_matrix_sqrt, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)

    if exit_code != 0:
        print("CG didn't converge with generalized_CG for fluctuating term ! Exitcode :", exit_code, flush=True)
    return fluctuating_map.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))



def solve_generalized_wiener_filter_term(param_dict, data_var, initial_guess, red_inverse_cov_matrix, red_inverse_noise, comm_world, language='Python', lmin=0, n_iter=8, limit_iter_cg=1000, tolerance=10**(-12), iter_alm=1000, tolerance_alm=10**(-10)):
    """ Solve Wiener filter term with formulation (1 + C^1/2 N^-1 C^1/2) for the left member, 
        and with the use of matrix square root to apply in harmonic space to the alms
    """
    
    assert (language == 'C') or (language == 'Python')

    assert red_inverse_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin
    if param_dict['nstokes'] != 1:
        assert data_var.shape[0] == param_dict['nstokes']

    red_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(np.linalg.pinv(red_inverse_cov_matrix))
    new_right_member_mat = np.zeros_like(red_inverse_cov_matrix)

    # new_right_member_mat[2:,...] = np.einsum('ljk,lkm->ljm', red_cov_sqrt,red_inverse_noise)[2:,...]
    new_right_member_mat = np.einsum('ljk,lkm->ljm', red_cov_sqrt,red_inverse_noise)
    # new_right_member_mat[2:,...] = red_inverse_noise[2:,...]
    # new_right_member_mat[2:,...] = red_inverse_cov_matrix[2:,...]


    new_matrix_cov = np.zeros_like(red_inverse_noise)
    # for ell in range(lmin, new_matrix_cov.shape[0]):
    #     new_matrix_cov[ell] = np.eye(param_dict['nstokes']) + np.linalg.multi_dot((red_cov_sqrt[ell],red_inverse_noise[ell],red_cov_sqrt[ell]))

    # red_eye = np.zeros_like(red_inverse_cov_matrix)
    # red_eye[2:,...] = np.repeat(np.eye(param_dict['nstokes']),red_inverse_cov_matrix.shape[0]-2).reshape((red_inverse_cov_matrix.shape[0]-2,param_dict['nstokes'],param_dict['nstokes']),order='F')
    
    # new_matrix_cov[2:,...] = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)[2:,...]
    new_matrix_cov = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)
    # new_matrix_cov[2:,...] = np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)[2:,...]
    # new_matrix_cov[2:,...] = red_eye[2:,...]

    if language == 'Python':
        right_member = (maps_x_reduced_matrix_generalized_sqrt(np.copy(data_var).reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_right_member_mat, lmin=lmin, n_iter=n_iter)).ravel()    
        func_left_term = lambda x : (maps_x_reduced_matrix_generalized_sqrt(x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), new_matrix_cov, lmin=lmin, n_iter=n_iter)).ravel()

    if language == 'C':
        raise Exception("C not added in this version")
        # CMB_map_Wiener_filter_0 = np.zeros(((param_dict["nstokes"],12*param_dict["nside"]**2)))
        
        # new_right_member_mat_sqrt = get_sqrt_reduced_matrix_from_matrix(new_right_member_mat)
        # new_matrix_cov_sqrt = get_sqrt_reduced_matrix_from_matrix(new_matrix_cov)

        # right_member = (mappraiser.apply_red_matrix_x_alm(param_dict, np.copy(data_var).reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), new_right_member_mat_sqrt, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)).ravel()
        # func_left_term = lambda x : (mappraiser.apply_red_matrix_x_alm(param_dict, x.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), new_matrix_cov_sqrt, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)).ravel()


    wiener_filter_term_z, number_iterations, exit_code = generalized_cg_from_func(initial_guess.ravel(), func_left_term, right_member, limit_iter_cg=limit_iter_cg, tolerance=tolerance)

    if language == 'Python':
        print("CG-Python-1 WF sqrt finished in ", number_iterations, "iterations !!")    
        wiener_filter_term = maps_x_reduced_matrix_generalized_sqrt(wiener_filter_term_z.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_cov_sqrt, lmin=lmin, n_iter=n_iter)
        # wiener_filter_term = wiener_filter_term_z.reshape((param_dict["nstokes"],12*param_dict["nside"]**2))
    
    # elif language == 'C':
        # print("CG-C-1 WF w/ iter finished in ", number_iterations, "iterations !!")
        # red_sqrt_covariance_matrix_sqrt = get_sqrt_reduced_matrix_from_matrix(red_cov_sqrt)
        # wiener_filter_term = mappraiser.apply_red_matrix_x_alm(param_dict, wiener_filter_term_z.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), np.copy(CMB_map_Wiener_filter_0), red_sqrt_covariance_matrix_sqrt, np.ones(12*param_dict['nside']**2), comm_world, iter_alm=iter_alm, error_alm=tolerance_alm)

    if exit_code != 0:
        print("CG didn't converge with WF ! Exitcode :", exit_code, flush=True)
    return wiener_filter_term.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))


class Gibbs_Sampling(object):
    def __init__(self, nside, lmax, nstokes, lmin=0, n_iter=8, language='Python', number_iterations_sampling=1000, limit_iter_cg=1000, tolerance_fluctuation=10**(-4), tolerance_PCG=10**(-12), iter_alm=1000, tolerance_alm=10**(-10), prior=0, option_ell_2=1):
        """ For the prior :
                0 : uniform prior
                1 : Jeffrey prior
            n_iter : Number of iterations for Python estimation of alms ; ignored if language='C'
            iter_alm : Number of iterations for C estimation of alms ; ignored if language='Python'
            tolerance_alm : Error for C estimation of alms ; ignored if language='C'
        """

        # Package params
        assert (language == 'C') or (language == 'Python')
        self.language=language

        # Problem parameters
        self.nside = nside
        self.lmax = lmax
        self.nstokes = nstokes
        self.lmin = lmin
        self.n_iter = n_iter # Number of iterations for Python estimation of alms ; ignored if language='C'
        self.iter_alm = iter_alm # Number of iterations for C estimation of alms ; ignored if language='Python'
        self.tolerance_alm = tolerance_alm # Error for C estimation of alms ; ignored if language='C'

        # CG parameters
        self.limit_iter_cg = limit_iter_cg
        self.tolerance_PCG = tolerance_PCG
        self.tolerance_fluctuation = tolerance_fluctuation

        # Sampling parameters
        self.number_iterations_sampling = number_iterations_sampling
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
        return int(np.ceil(self.nstokes**2/2) + np.floor(self.nstokes/2))
    
    def constrained_map_realization(self, initial_map, red_inverse_covariance_matrix, red_inverse_noise, comm_world):
        """ Constrained realization step """
        
        assert red_inverse_covariance_matrix.shape[0] == self.lmax + 1 - self.lmin
        assert red_inverse_noise.shape[0] == self.lmax + 1 - self.lmin
        if self.nstokes != 1:
            assert initial_map.shape[0] == self.nstokes
        
        param_dict = {'nside':self.nside, 'lmax':self.lmax, 'nstokes':self.nstokes, 'number_correlations':self.number_correlations}
        
        initial_guess = np.zeros((self.nstokes, self.npix))

        map_white_noise_xi = []
        map_white_noise_chi = []
        # print("Test 0a ! lmin", self.lmin, flush=True)
        fluctuating_map = get_fluctuating_term_maps(param_dict, np.copy(initial_guess), red_inverse_covariance_matrix, red_inverse_noise, comm_world, map_white_noise_xi, map_white_noise_chi, language=self.language, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_fluctuation, iter_alm=self.iter_alm, tolerance_alm=self.tolerance_alm)
        # print("Test 0b !", flush=True)
        
        wiener_filter_term = solve_generalized_wiener_filter_term(param_dict, initial_map, np.copy(initial_guess), red_inverse_covariance_matrix, red_inverse_noise, comm_world, language=self.language, lmin=self.lmin, n_iter=self.n_iter, limit_iter_cg=self.limit_iter_cg, tolerance=self.tolerance_PCG, iter_alm=self.iter_alm, tolerance_alm=self.tolerance_alm)
        # path_map_sample = "/Users/mag/Documents/PHD1Y/Space_Work/Inference_Sampling/map_files/iter_tests/"
        # outname_fluct = "Intermediate_map_fluct_test1.fits"
        # outname_WF = "Intermediate_map_WF_test1.fits"
        # print("~~~~ Record test in :", path_map_sample, outname_fluct, outname_WF)
        # hp.write_map(path_map_sample+outname_fluct, fluctuating_map, overwrite=True)
        # hp.write_map(path_map_sample+outname_WF, wiener_filter_term, overwrite=True)
        # print("Test 0c !", flush=True)
        return wiener_filter_term + fluctuating_map
    
    def sample_covariance(self, pixel_maps):
        """ Power spectrum sampling """
        c_ells_Wishart = get_cell_from_map(pixel_maps, lmax=self.lmax, n_iter=self.n_iter)
        return get_inverse_wishart_sampling_from_c_ells(c_ells_Wishart, lmax=self.lmax, q_prior=self.prior, l_min=self.lmin, option_ell_2=self.option_ell_2)#[self.lmin:,...]

    def sample_sky(self, initial_map, c_ells_array, red_inverse_noise, comm_world, file_ver="", dir_path='', record_test_bool=False):

        c_ell_sampled = np.copy(c_ells_array)
        pixel_maps_sampled = np.copy(initial_map)

        if self.nstokes != 1:
            assert initial_map.shape[0] == self.nstokes

        # diff_log_proba = np.zeros(self.number_iterations_sampling)

        # max_proba = 0
        # maximum_posterior_c_ells = np.zeros_like(c_ell_sampled)
        # maximum_posterior_maps = np.zeros((self.nstokes, self.npix))

        all_maps = np.zeros((self.number_iterations_sampling+1, self.nstokes, self.npix))
        # all_samples = np.zeros((self.number_iterations_sampling, self.number_correlations, self.lmax+1-self.lmin))
        all_samples = np.zeros((self.number_iterations_sampling+1, self.number_correlations, self.lmax+1))

        all_maps[0,...] = pixel_maps_sampled
        all_samples[0,...] = c_ell_sampled

        if red_inverse_noise.shape[0] == self.lmax+1:
            red_inverse_noise = red_inverse_noise[self.lmin:,...]
        
        # print("Test 0a :", c_ell_sampled.shape, c_ell_sampled[0].mean(), c_ell_sampled[1].mean(), c_ell_sampled[2].mean())#, c_ell_sampled[3].mean(), c_ell_sampled[4].mean())
        # print("Test 0b :", c_ell_sampled.shape, c_ell_sampled[0][c_ell_sampled[0]<=0], c_ell_sampled[1][c_ell_sampled[1]<=0], len(c_ell_sampled[2][c_ell_sampled[2]<0]), len(c_ell_sampled[2][c_ell_sampled[2]>0]))

        for iteration in range(self.number_iterations_sampling):
            print("### Start Iteration n°", iteration, flush=True)

            # if record_test_bool:# and iteration < 100:
            #     # file_ver = 'SkySamplingv1a' # Firsts tests sampling sky with noise_level 10**(-7)
            #     dir_path = '/Users/mag/Documents/PHD1Y/Space_Work/Inference_Sampling/map_files/iter_tests/'
            #     outname_map = 'Map_ver{}_iter{}_nside{}_{}.fits'.format(file_ver, iteration, self.nside, self.lmax)
            #     outname_c_ell = 'Cell_ver{}_iter{}_nside{}_{}.npy'.format(file_ver, iteration, self.nside, self.lmax)
            #     if self.nstokes != 2:
            #         hp.write_map(dir_path+outname_map, pixel_maps_sampled, overwrite=True)
            #     else:
            #         hp.write_map(dir_path+outname_map, [np.zeros_like(pixel_maps_sampled[0]), pixel_maps_sampled[0], pixel_maps_sampled[1]], overwrite=True)
            #     np.save(dir_path+outname_c_ell, c_ell_sampled)
            #     print("Save Iteration n°", iteration, 'in', dir_path, outname_map, outname_c_ell, flush=True)

            # red_inverse_covariance_matrix = get_inverse_reduced_matrix_from_c_ell(c_ell_sampled[:,self.lmin:], lmin=self.lmin)
            red_inverse_covariance_matrix = get_inverse_reduced_matrix_from_c_ell(c_ell_sampled, lmin=self.lmin)
        #    inverse_covariance_matrix = np.linalg.inv(get_covariance_matrix_from_c_ell(c_ell_sampled))
        #    red_inverse_covariance_matrix = get_inverse_reduced_matrix_from_c_ell(c_ell_sampled, lmin=l_min, lmax=self.lmax+1)
            
        #    c_ell_sampled = c_ell_sampled[:,:self.lmax]
            # print("Iteration-0 n°", iteration, flush=True)
            # print("Test 0 !", flush=True)
            # pixel_maps_sampled = self.constrained_map_realization(pixel_maps_sampled, red_inverse_covariance_matrix, red_inverse_noise, comm_world)
            pixel_maps_sampled = self.constrained_map_realization(initial_map, red_inverse_covariance_matrix, red_inverse_noise, comm_world)
            # Constrained realization step
                

        #    c_ell_sampled = self.sample_c_ell(pixel_maps_sampled, n_iter=n_iter, l_min=l_min)
            red_cov_mat_sampled = self.sample_covariance(pixel_maps_sampled)
            # print("Test 40", red_cov_mat_sampled[self.lmin:10,...].shape)
            # print("Test 4", red_cov_mat_sampled[self.lmin:10,...], np.linalg.eigh(red_cov_mat_sampled[self.lmin:])[0])
            # assert np.all(np.linalg.eigh(red_cov_mat_sampled[self.lmin:])[0] > 0)
            all_eigenvalues = np.linalg.eigh(red_cov_mat_sampled[self.lmin:])[0]
            assert np.all(all_eigenvalues[np.abs(np.linalg.eigh(red_cov_mat_sampled[self.lmin:])[0])>10**(-15)]>0)

            # c_ell_sampled = np.zeros_like(c_ell_sampled)
            # c_ell_sampled[:,self.lmin:] = get_c_ells_from_red_covariance_matrix(red_cov_mat_sampled)
            c_ell_sampled = get_c_ells_from_red_covariance_matrix(red_cov_mat_sampled)
            # Power sampling step

            all_maps[iteration+1,...] = pixel_maps_sampled
            all_samples[iteration+1,...] = c_ell_sampled
            # average_proba = 0
            # sigma_ell = get_cell_from_map(pixel_maps_sampled, lmax=self.lmax, n_iter=self.n_iter)
            # for i in range(self.nstokes):
            #     sigma_ell[i] *= 2*np.arange(self.lmax+1) + 1
            # sigma_ell_param_matrix_Wishart = get_reduced_matrix_from_c_ell(sigma_ell)
            # # average_proba_array = get_generalized_proba_inverse_wishart_distribution(red_cov_mat_sampled[self.lmin:,...], sigma_ell_param_matrix_Wishart[self.lmin:,...], 2*np.arange(self.lmin,self.lmax+1) + 1)
            # # print("Test", average_proba_array.shape, flush=True)
            # for ell in range(max(self.lmin,2), self.lmax+1):                
            #     average_proba += get_proba_inverse_wishart_distribution(red_cov_mat_sampled[ell,...], sigma_ell_param_matrix_Wishart[ell,...], 2*ell+1) /(self.lmax+1-max(self.lmin,2))

            # if average_proba > max_proba:
            #     max_proba = average_proba
            #     maximum_posterior_c_ells = c_ell_sampled
            #     maximum_posterior_maps = pixel_maps_sampled 
            #     print("New maximum posterior for ell=", ell, flush=True)

            if iteration%50 == 0:
                print("### Iteration n°", iteration, flush=True)
        return all_maps, all_samples
        # return maximum_posterior_maps, maximum_posterior_c_ells
