import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import healpy as hp
import astropy.io.fits as fits
import camb
# from matplotlib.colors import LogNorm, SymLogNorm

import non_parametric_ML_compsep as blindcp


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
