import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import healpy as hp
import astropy.io.fits as fits
import camb
# from matplotlib.colors import LogNorm, SymLogNorm

import micmac as blindcp


def generate_power_spectra_CAMB(Nside,  r=10**(-2), Alens=1, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, ns=0.965, lens_potential_accuracy=1, nt=0, ntrun=0, type_power='total', typeless_bool=False):
    """
    Return [Cl^TT, Cl^EE, Cl^BB, Cl^TE]
    """
    lmax = 2*Nside
    # pars = camb.CAMBparams(max_l_tensor=lmax, parameterization='tensor_param_indeptilt')
    pars = camb.CAMBparams(max_l_tensor=lmax)
    pars.WantTensors = True

    pars.Accuracy.AccurateBB = True
    pars.Accuracy.AccuratePolarization = True
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, Alens=Alens)
    pars.InitPower.set_params(As=2e-9, ns=ns, r=r, parameterization='tensor_param_indeptilt', nt=nt, ntrun=ntrun)
    pars.max_eta_k_tensor = lmax + 100  # 15000  # 100

    # pars.set_cosmology(H0=H0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)

    print("Calculating spectra from CAMB !")
    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lmax=lmax)    
    if typeless_bool:
        return powers
    return powers[type_power]


def get_all_analytical_WF_term_maps_v2(param_dict, data_var, red_cov_matrix, red_inverse_noise, lmin=0, n_iter=8):
    """ 
        Solve Wiener filter term with formulation (C^-1 + N^-1) for the left member, 
        and with the use of matrix square root to apply in harmonic space to the alms
    """

    assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin

    red_cov_sqrt = blindcp.get_sqrt_reduced_matrix_from_matrix(red_cov_matrix)

    new_matrix_cov_0 = np.linalg.pinv(red_cov_matrix) + red_inverse_noise
    new_matrix_cov_1 = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)

    right_member_0_rhs = np.einsum('ljk,lkm->ljm', np.linalg.pinv(new_matrix_cov_0), red_inverse_noise)

    right_member_1_rhs = np.einsum('lkj,ljm,lmn,lno->lko', red_cov_sqrt, np.linalg.pinv(new_matrix_cov_1), red_cov_sqrt, red_inverse_noise)

    # WF_solution_0 = blindcp.maps_x_reduced_matrix_generalized_sqrt(data_var.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs, lmin=lmin, n_iter=n_iter).ravel()
    # WF_solution_1 = blindcp.maps_x_reduced_matrix_generalized_sqrt(data_var.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs, lmin=lmin, n_iter=n_iter).ravel()
    WF_solution_0 = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(data_var.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs, lmin=lmin, n_iter=n_iter).ravel()
    WF_solution_1 = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(data_var.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs, lmin=lmin, n_iter=n_iter).ravel()
    print("Exact-Python-0 WF sqrt exact computed for WF term !!")    
    
    return WF_solution_0.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2)), WF_solution_1.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))


# def get_all_analytical_WF_term_maps_v3sqrt(param_dict, data_var, red_cov_matrix, red_inverse_noise, lmin=0, n_iter=8):
#     """ 
#         Solve Wiener filter term with formulation (C^-1 + N^-1) for the left member, 
#         and with the use of matrix square root to apply in harmonic space to the alms
#     """

#     assert red_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
#     assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin

#     red_cov_sqrt = blindcp.get_sqrt_reduced_matrix_from_matrix(red_cov_matrix)

#     new_matrix_cov_0 = np.linalg.pinv(red_cov_matrix) + red_inverse_noise
#     new_matrix_cov_1 = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)

#     right_member_0_rhs = np.einsum('ljk,lkm->ljm', np.linalg.pinv(new_matrix_cov_0), red_inverse_noise)

#     right_member_1_rhs = np.einsum('lkj,ljm,lmn,lno->lko', red_cov_sqrt, np.linalg.pinv(new_matrix_cov_1), red_cov_sqrt, red_inverse_noise)

#     WF_solution_0 = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(data_var.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs, lmin=lmin, n_iter=n_iter).ravel()

#     WF_solution_1 = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(data_var.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs, lmin=lmin, n_iter=n_iter).ravel()
#     print("Exact-Python-0 WF sqrt exact computed for WF term !!")    
    
#     return WF_solution_0.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2)), WF_solution_1.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))


def get_all_analytical_fluctuating_term_maps_v2(param_dict, red_inverse_cov_matrix, red_inverse_noise, map_white_noise_xi, map_white_noise_chi, lmin=0, n_iter=8):
    """ 
        Solve Wiener filter term with formulation (C^-1 + N^-1) for the left member, 
        and with the use of matrix square root to apply in harmonic space to the alms
    """
    assert red_inverse_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin

    red_cov_sqrt_inv = blindcp.get_sqrt_reduced_matrix_from_matrix(red_inverse_cov_matrix)
    red_cov_sqrt = blindcp.get_sqrt_reduced_matrix_from_matrix(np.linalg.pinv(red_inverse_cov_matrix))
    red_inv_noise_sqrt = blindcp.get_sqrt_reduced_matrix_from_matrix(red_inverse_noise)

    new_matrix_cov_0 = red_inverse_cov_matrix + red_inverse_noise
    new_matrix_cov_1 = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)

    # if len(map_white_noise_xi) == 0:
    #     map_white_noise_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(nside), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    # if len(map_white_noise_chi) == 0:
    #     map_white_noise_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(nside), size=(param_dict["nstokes"],12*param_dict["nside"]**2))

    right_member_0_rhs_a = np.einsum('ljk,lkm->ljm', np.linalg.pinv(new_matrix_cov_0), red_cov_sqrt_inv)
    right_member_0_rhs_b = np.einsum('ljk,lkm->ljm', np.linalg.pinv(new_matrix_cov_0), red_inv_noise_sqrt)

    right_member_1_rhs_a = np.einsum('lkj,ljm->lkm', red_cov_sqrt, np.linalg.pinv(new_matrix_cov_1))
    right_member_1_rhs_b = np.einsum('lij,ljk,lkm,lmn->lin', red_cov_sqrt, np.linalg.pinv(new_matrix_cov_1), red_cov_sqrt, red_inv_noise_sqrt)

    right_member_0_a = blindcp.maps_x_reduced_matrix_generalized_sqrt(map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs_a, lmin=lmin, n_iter=n_iter)
    right_member_0_b = blindcp.maps_x_reduced_matrix_generalized_sqrt(map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs_b, lmin=lmin, n_iter=n_iter)

    right_member_1_a = blindcp.maps_x_reduced_matrix_generalized_sqrt(map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs_a, lmin=lmin, n_iter=n_iter)
    right_member_1_b = blindcp.maps_x_reduced_matrix_generalized_sqrt(map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs_b, lmin=lmin, n_iter=n_iter)

    fluct_solution_0 = (right_member_0_a + right_member_0_b).ravel()
    fluct_solution_1 = (right_member_1_a + right_member_1_b).ravel()

    print("Exact-Python-0 Fluct sqrt exact computed for fluctuating term !!")    

    return fluct_solution_0.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2)), fluct_solution_1.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))

def get_all_analytical_fluctuating_term_maps_v3sqrt(param_dict, red_inverse_cov_matrix, red_inverse_noise, map_white_noise_xi, map_white_noise_chi, lmin=0, n_iter=8):
    """ 
        Solve Wiener filter term with formulation (C^-1 + N^-1) for the left member, 
        and with the use of matrix square root to apply in harmonic space to the alms
    """
    assert red_inverse_cov_matrix.shape[0] == param_dict['lmax'] + 1 - lmin
    assert red_inverse_noise.shape[0] == param_dict['lmax'] + 1 - lmin

    red_cov_sqrt_inv = blindcp.get_sqrt_reduced_matrix_from_matrix(red_inverse_cov_matrix)
    red_cov_sqrt = blindcp.get_sqrt_reduced_matrix_from_matrix(np.linalg.pinv(red_inverse_cov_matrix))
    red_inv_noise_sqrt = blindcp.get_sqrt_reduced_matrix_from_matrix(red_inverse_noise)

    new_matrix_cov_0 = red_inverse_cov_matrix + red_inverse_noise
    new_matrix_cov_1 = np.eye(param_dict['nstokes']) + np.einsum('ljk,lkm,lmn->ljn', red_cov_sqrt,red_inverse_noise,red_cov_sqrt)

    # if len(map_white_noise_xi) == 0:
    #     map_white_noise_xi = np.random.normal(loc=0, scale=1/hp.nside2resol(nside), size=(param_dict["nstokes"],12*param_dict["nside"]**2))
    # if len(map_white_noise_chi) == 0:
    #     map_white_noise_chi = np.random.normal(loc=0, scale=1/hp.nside2resol(nside), size=(param_dict["nstokes"],12*param_dict["nside"]**2))

    right_member_0_rhs_a = np.einsum('ljk,lkm->ljm', np.linalg.pinv(new_matrix_cov_0), red_cov_sqrt_inv)
    right_member_0_rhs_b = np.einsum('ljk,lkm->ljm', np.linalg.pinv(new_matrix_cov_0), red_inv_noise_sqrt)

    right_member_1_rhs_a = np.einsum('lkj,ljm->lkm', red_cov_sqrt, np.linalg.pinv(new_matrix_cov_1))
    right_member_1_rhs_b = np.einsum('lij,ljk,lkm,lmn->lin', red_cov_sqrt, np.linalg.pinv(new_matrix_cov_1), red_cov_sqrt, red_inv_noise_sqrt)

    right_member_0_a = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs_a, lmin=lmin, n_iter=n_iter)
    right_member_0_b = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_0_rhs_b, lmin=lmin, n_iter=n_iter)

    right_member_1_a = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(map_white_noise_xi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs_a, lmin=lmin, n_iter=n_iter)
    right_member_1_b = blindcp.maps_x_reduced_matrix_generalized_sqrt_sqrt(map_white_noise_chi.reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), right_member_1_rhs_b, lmin=lmin, n_iter=n_iter)

    fluct_solution_0 = (right_member_0_a + right_member_0_b).ravel()
    fluct_solution_1 = (right_member_1_a + right_member_1_b).ravel()

    print("Exact-Python-0 Fluct sqrt exact computed for fluctuating term !!")    

    return fluct_solution_0.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2)), fluct_solution_1.reshape((param_dict["nstokes"], 12*param_dict["nside"]**2))

def personnalized_synfast(param_dict, red_matrix, random_map=[], lmin=2, n_iter=8):
    if len(random_map) == 0:
        print("Generating random map !")
        random_map = np.random.normal(loc=0, scale=1/hp.nside2resol(param_dict["nside"]), size=(param_dict["nstokes"], 12*param_dict["nside"]**2))
    
    return blindcp.maps_x_reduced_matrix_generalized_sqrt(np.copy(random_map).reshape((param_dict["nstokes"],12*param_dict["nside"]**2)), red_matrix, lmin=lmin, n_iter=n_iter)

def hp_synfast(param_dict, red_matrix, lmin=2, n_iter=8):
    c_ell_red = blindcp.get_c_ells_from_red_covariance_matrix(red_matrix)
    c_ell_to_use = np.zeros((6, param_dict['lmax']+1))
    if red_matrix.shape[1] == 2:
        c_ell_to_use[1:3, lmin:] = c_ell_red[:2]
        c_ell_to_use[4, lmin:] = c_ell_red[2]

    else:
        raise Exception("Error")

    return hp.synfast(c_ell_to_use, param_dict['nside'], lmax=param_dict['lmax'], new=True)




def personalized_get_inverse_wishart_sampling_from_c_ells(sigma_ell, lmax=None, q_prior=0, n_iter=0, l_min=0, option_ell_2=0):
    """ Assume sigma_ell to be with all ells
        
        Also assumes the monopole and dipole to be 0

        q_prior = 0 : uniform prior
        q_prior = 1 : Jeffrey prior

    """

    # nstokes = len(pixel_maps.shape)
    # if nstokes > 1:
    #     nstokes = pixel_maps.shape[0]

    if len(sigma_ell.shape) == 1:
        nstokes == 1
    elif sigma_ell.shape[0] == 6:
        nstokes = 3
    elif sigma_ell.shape[0] == 3:
        nstokes = 2


    # sigma_ell = hp.anafast(pixel_maps, lmax=lmax, iter=n_iter)

    # number_correl = c_ells_observed.shape[0]
    # print("Test", nstokes, number_correl)
    # c_ells_observed = np.vstack((np.zeros((number_correl,1)), c_ells_observed))
    # c_ells_observed = np.hstack((np.zeros((number_correl,1)), c_ells_observed))
    for i in range(nstokes):
        sigma_ell[i] *= 2*np.arange(lmax+1) + 1

    lmin = l_min
    # invert_parameter_Wishart = get_inverse_reduced_matrix_from_red_matrix(get_reduced_matrix_from_c_ell(sigma_ell))
    # invert_parameter_Wishart = np.linalg.pinv(pss.get_reduced_matrix_from_c_ell(sigma_ell))[l_min:,...]
    invert_parameter_Wishart = np.linalg.pinv(blindcp.get_reduced_matrix_from_c_ell(sigma_ell))
    # print("Test size red:", invert_parameter_Wishart.shape, "sigma_ell", sigma_ell.shape)
    assert invert_parameter_Wishart.shape[0] == lmax + 1 #- lmin
    sampling_Wishart = np.zeros_like(invert_parameter_Wishart)

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
            # new_cov_matrix[ell] = np.linalg.inv(np.dot(sample_gaussian.T,sample_gaussian))
            # sampling_Wishart[ell] = np.linalg.pinv(np.dot(sample_gaussian.T,sample_gaussian))
            sampling_Wishart[ell_2] = np.dot(sample_gaussian.T,sample_gaussian)

        # Option sampling separately TE and B
        elif option_ell_2 == 1:
            print("Sampling separately TE and B for ell=2 !", flush=True)
            invert_parameter_Wishart_2 = np.zeros((nstokes,nstokes))
            reduced_matrix_2 = blindcp.get_reduced_matrix_from_c_ell(sigma_ell)[ell_2]
            invert_parameter_Wishart_2[:nstokes-1, :nstokes-1] = np.linalg.pinv(reduced_matrix_2[:nstokes-1,:nstokes-1])
            invert_parameter_Wishart_2[nstokes-1, nstokes-1] = 1/reduced_matrix_2[nstokes-1,nstokes-1]
            # sample_gaussian_TE = np.random.multivariate_normal(np.zeros(nstokes-1), invert_parameter_Wishart[ell_2][:nstokes-1, :nstokes-1], size=(2*ell_2 - (nstokes-1)))
            # sample_gaussian_B = np.random.normal(loc=0, scale=invert_parameter_Wishart[ell_2][nstokes-1, nstokes-1], size=(2*ell_2 - 1))
            sample_gaussian_TE = np.random.multivariate_normal(np.zeros(nstokes-1), invert_parameter_Wishart_2[:nstokes-1, :nstokes-1], size=(2*ell_2 - (nstokes-1)))
            sample_gaussian_B = np.random.normal(loc=0, scale=invert_parameter_Wishart_2[nstokes-1, nstokes-1], size=(2*ell_2 - 1))
            sampling_Wishart[ell_2][:nstokes-1,:nstokes-1] = np.dot(sample_gaussian_TE.T,sample_gaussian_TE)
            sampling_Wishart[ell_2][nstokes-1,nstokes-1] = np.dot(sample_gaussian_B.T,sample_gaussian_B)
        
        lmin = 3


    # print("### TEST : lmax+1", lmax+1, "lmin", max(lmin,2), "nstokes", nstokes, flush=True)
    for ell in range(max(lmin,2),lmax+1):
        # number_vectors = 2*ell - nstokes + 2*q_prior
        # mean = np.zeros(nstokes)
        # sample_gaussian = np.random.multivariate_normal(np.diag(cov_matrix[ell]), cov_matrix[ell], size=(2*ell - nstokes + 2*q_prior))
        # sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell-l_min], size=(2*ell - nstokes + 2*q_prior))
        sample_gaussian = np.random.multivariate_normal(np.zeros(nstokes), invert_parameter_Wishart[ell], size=(2*ell - nstokes + 2*q_prior))
        # if ell >= lmax-1 or ell == 2:
        #     print("ell", ell, cov_matrix[ell], np.dot(sample_gaussian.T,sample_gaussian))
        #     print("rank", ell, np.linalg.matrix_rank(cov_matrix[ell]), np.linalg.matrix_rank(np.dot(sample_gaussian.T,sample_gaussian)))
        # new_cov_matrix[ell] = np.linalg.inv(np.dot(sample_gaussian.T,sample_gaussian))
        # sampling_Wishart[ell] = np.linalg.pinv(np.dot(sample_gaussian.T,sample_gaussian))
        # sampling_Wishart[ell-lmin] = np.dot(sample_gaussian.T,sample_gaussian)
        sampling_Wishart[ell] = np.dot(sample_gaussian.T,sample_gaussian)
        # print("## ell ", ell, end='')

    # return get_inverse_reduced_matrix_from_red_matrix(sampling_Wishart)
    return np.linalg.pinv(sampling_Wishart)

def get_all_invW(sigma_ell, number_samples=1, lmax=None, q_prior=0, n_iter=0, l_min=0, option_ell_2=0):
    # nstokes = len(pixel_maps.shape)
    # if nstokes > 1:
    #     nstokes = pixel_maps.shape[0]
    if len(sigma_ell.shape) == 1:
        nstokes == 1
    elif sigma_ell.shape[0] == 6:
        nstokes = 3
    elif sigma_ell.shape[0] == 3:
        nstokes = 2

    # result_covariance = np.zeros((number_samples, lmax+1-l_min, nstokes, nstokes))
    result_covariance = np.zeros((number_samples, lmax+1, nstokes, nstokes))
    for nb_sample in range(number_samples):
        result_covariance[nb_sample] = personalized_get_inverse_wishart_sampling_from_c_ells(np.copy(sigma_ell), lmax, q_prior, n_iter, l_min, option_ell_2)
    return result_covariance
