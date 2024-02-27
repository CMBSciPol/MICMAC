import os, sys, time
import numpy as np
import healpy as hp
import scipy.linalg

def get_reduced_matrix_from_c_ell(c_ells_input):
    """ Expect c_ells_input to be sorted as TT, EE, BB, TE, TB, EB if 6 spectra are given

        Generate covariance matrix from c_ells assuming it's block diagonal
    """
    c_ells_array = np.copy(c_ells_input)
    n_correlations = c_ells_array.shape[0]
    assert n_correlations == 1 or n_correlations == 3 or n_correlations == 6
    lmax_p1 = c_ells_array.shape[1]
    if n_correlations == 1:
        nstokes = 1
    elif n_correlations == 3:
        nstokes = 2
        # c_ells_array = np.vstack((c_ells_array, np.zeros(lmax)))
        # n_correlations = 3
    # elif n_correlations > 3:
    elif n_correlations == 4 or n_correlations == 6 :
        nstokes = 3
        if n_correlations != 6:
            for i in range(6 - n_correlations):
                c_ells_array = np.vstack((c_ells_array, np.zeros(lmax_p1)))
            n_correlations = 6
    else :
        raise Exception("C_ells must be given as TT for temperature only ; EE, BB, EB for polarization only ; TT, EE, BB, TE, (TB, EB) for both temperature and polarization")

    reduced_matrix = np.zeros((lmax_p1,nstokes,nstokes))

    for i in range(nstokes):
        reduced_matrix[:,i,i] =  c_ells_array[i,:]
    
    # for j in range(n_correlations-nstokes):
    if n_correlations > 1:
        reduced_matrix[:,0,1] =  c_ells_array[nstokes,:]
        reduced_matrix[:,1,0] =  c_ells_array[nstokes,:]

    if n_correlations == 6:
        # reduced_matrix[:,0,2] =  c_ells_array[4,:]
        # reduced_matrix[:,2,0] =  c_ells_array[4,:]

        # reduced_matrix[:,1,2] =  c_ells_array[5,:]
        # reduced_matrix[:,2,1] =  c_ells_array[5,:]

        reduced_matrix[:,0,2] =  c_ells_array[5,:]
        reduced_matrix[:,2,0] =  c_ells_array[5,:]

        reduced_matrix[:,1,2] =  c_ells_array[4,:]
        reduced_matrix[:,2,1] =  c_ells_array[4,:]

    return reduced_matrix

def get_c_ells_from_red_covariance_matrix(red_cov_mat):
    """ Retrieve c_ell from red_cov_mat, which depending of nstokes will give :
            TT
            EE, BB, EB
            TT, EE, BB, TE, EB, TB
    """
    
    lmax = red_cov_mat.shape[0]
    nstokes = red_cov_mat.shape[1]

    n_correl = int(np.ceil(nstokes**2/2) + np.floor(nstokes/2))
    c_ells = np.zeros((n_correl, lmax))

    for i in range(nstokes):
        c_ells[i,:] = red_cov_mat[:,i,i]
    if nstokes > 1:
        c_ells[nstokes,:] = red_cov_mat[:,0,1]
        if nstokes == 3:
            # c_ells[nstokes+1,:] = red_cov_mat[:,0,2]
            # c_ells[nstokes+2,:] = red_cov_mat[:,1,2]
            c_ells[nstokes+2,:] = red_cov_mat[:,0,2]
            c_ells[nstokes+1,:] = red_cov_mat[:,1,2]
    return c_ells

def get_inverse_reduced_matrix_from_c_ell(c_ells_input, lmin=0):
    return np.linalg.pinv(get_reduced_matrix_from_c_ell(c_ells_input))[lmin:,...]

def get_inverse_reduced_matrix_from_red_matrix(red_matrix):
    return np.linalg.pinv(red_matrix)

# def get_cholesky_reduced_matrix_from_matrix(red_matrix, lmin=0):
#     """ Return L Cholesky decomposition
#     """
#     # if lmax == -1:
#     lmax = red_matrix.shape[0]
#     nstokes = red_matrix.shape[1]

#     reduced_cholesky_decomposition = np.zeros_like(red_matrix)
#     if lmin < 2:
#         for ell in range(lmin, 2):
#             try :
#                 reduced_cholesky_decomposition[ell,:,:] = np.linalg.cholesky(red_matrix[ell,:,:])
#             except:
#                 reduced_cholesky_decomposition[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
#                 eigvals, eigevcs = np.linalg.eig(red_matrix[ell,:,:])
#                 # if not(np.allclose(eigvals, np.zeros(nstokes))):
#                 if np.any(eigvals<0) or not(np.allclose(eigvals, np.zeros(nstokes))):
#                     print("Cannot get Cholesky decomposition of monopole of dipole :", ell, flush=True)
#     for ell in range(max(2,lmin),lmax):
#         # print("ell :", ell, "matrix :", red_matrix[ell,:,:])
#         reduced_cholesky_decomposition[ell,:,:] = np.linalg.cholesky(red_matrix[ell,:,:])
#     return reduced_cholesky_decomposition

def get_sqrt_reduced_matrix_from_matrix(red_matrix, tolerance=10**(-15)):
    """ Return L square root matrix
    """
    # if lmax == -1:
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    reduced_sqrtm = np.zeros_like(red_matrix)
    # if np.any(np.iscomplex(red_matrix)):
    #     print("COMPLEX ELEMENT IN RED MATRIX !!!", flush=True)
    # for ell in range(lmin,lmax+lmin):
    for ell in range(red_matrix.shape[0]):
        # if np.any(np.iscomplex(scipy.linalg.sqrtm(red_matrix[ell,:,:]))):
        #     print("COMPLEX ELEMENT IN SQRT RED MATRIX FOR ELL = {} !!!".format(ell), flush=True)
        # reduced_sqrtm[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
        eigvals, eigvect = np.linalg.eigh(red_matrix[ell,:,:])

        try:
            inv_eigvect = np.linalg.pinv(eigvect)
        except:
            raise Exception("Error for ell=",ell, "eigvals", eigvals, "eigvect", eigvect, "red_matrix", red_matrix[ell,:,:])

        if not(np.all(eigvals>0)) and (np.abs(eigvals[eigvals<0]) > tolerance):
            raise Exception("Covariance matrix not consistent with a negative eigval for ell=",ell, "eigvals", eigvals, "eigvect", eigvect, "red_matrix", red_matrix[ell,:,:])

        reduced_sqrtm[ell] = np.einsum('jk,km,m,mn->jn', eigvect, np.eye(nstokes), np.sqrt(np.abs(eigvals)), inv_eigvect)
    return reduced_sqrtm

def get_sqrt_reduced_matrix_from_matrix_alternative(red_matrix):
    """ Return L square root matrix
    """
    reduced_sqrtm = np.zeros_like(red_matrix)
    for ell in range(red_matrix.shape[0]):
        reduced_sqrtm[ell,:,:] = np.real(scipy.linalg.sqrtm(red_matrix[ell,:,:]))
    return reduced_sqrtm

def get_cell_from_map(pixel_maps, lmax, n_iter=8):
    
    if len(pixel_maps.shape) == 1:
        nstokes = 1
    else:
        nstokes = pixel_maps.shape[0]
    
    if nstokes == 2:
        pixel_maps_for_Wishart = np.vstack((np.zeros_like(pixel_maps[0]), pixel_maps))
            # print("Test 5 :", pixel_maps_for_Wishart.shape, pixel_maps_for_Wishart[0].mean(), pixel_maps_for_Wishart[1].mean(), pixel_maps_for_Wishart[2].mean())
    else:
        pixel_maps_for_Wishart = pixel_maps

    c_ells_Wishart = hp.anafast(pixel_maps_for_Wishart, lmax=lmax, iter=n_iter)

    if nstokes == 2:
        polar_indexes = np.array([1,2,4])
        c_ells_Wishart = c_ells_Wishart[polar_indexes]
    return c_ells_Wishart


def maps_x_reduced_matrix_generalized_sqrt_sqrt(maps_TQU_input, red_matrix_sqrt, lmin, n_iter=8):
    # if lmax == -1:
    lmax = red_matrix_sqrt.shape[0] - 1 + lmin
    nstokes = red_matrix_sqrt.shape[1]
    all_params = int(np.where(nstokes > 1, 3, 1))

    if len(maps_TQU_input.shape) == 1:
        nside = int(np.sqrt(len(maps_TQU_input)/12))
    else:
        nside = int(np.sqrt(len(maps_TQU_input[0])/12))
    
    # red_sqrt_decomp = np.zeros_like(red_matrix)
    red_sqrt_decomp = np.zeros((lmax+1,all_params,all_params))
    if nstokes != 1:
        red_sqrt_decomp[lmin:,3-nstokes:,3-nstokes:] = red_matrix_sqrt
    else:
        red_sqrt_decomp[lmin:,...] = red_matrix_sqrt

    if maps_TQU_input.shape[0] == 2:
        maps_TQU = np.vstack((np.zeros_like(maps_TQU_input[0]),np.copy(maps_TQU_input)))
    else:
        maps_TQU = np.copy(maps_TQU_input)

    alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    alms_output = np.zeros_like(alms_input)

    for i in range(all_params):
        alms_j = np.zeros_like(alms_input[i])
        for j in range(all_params):
            alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:,i,j], inplace=False)
        alms_output[i] = np.copy(alms_j)
    maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3-nstokes:,...]
    return maps_output

# def maps_x_reduced_matrix_generalized_sqrt(maps_TQU_input, red_matrix, lmin=0, n_iter=0):
#     # if lmax == -1:
#     lmax = red_matrix.shape[0] - 1 + lmin
#     nstokes = red_matrix.shape[1]
#     all_params = int(np.where(nstokes > 1, 3, 1))

#     if len(maps_TQU_input.shape) == 1:
#         nside = int(np.sqrt(len(maps_TQU_input)/12))
#     else:
#         nside = int(np.sqrt(len(maps_TQU_input[0])/12))
    
#     # red_sqrt_decomp = np.zeros_like(red_matrix)
#     red_sqrt_decomp = np.zeros((lmax+1,all_params,all_params))
#     if nstokes != 1:
#         red_sqrt_decomp[lmin:,3-nstokes:,3-nstokes:] = get_sqrt_reduced_matrix_from_matrix(red_matrix)
#         # red_cholesky_decomp = get_cholesky_reduced_matrix_from_matrix(red_matrix, lmin=lmin)
#     else:
#         red_sqrt_decomp[lmin:,...] = get_sqrt_reduced_matrix_from_matrix(red_matrix)
#         # red_cholesky_decomp = get_cholesky_reduced_matrix_from_matrix(red_matrix, lmin=lmin)

#     if maps_TQU_input.shape[0] == 2:
#         maps_TQU = np.vstack((np.zeros_like(maps_TQU_input[0]),maps_TQU_input))
#     else:
#         maps_TQU = np.copy(maps_TQU_input)

#     alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
#     # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
#     alms_output = np.zeros_like(alms_input)

#     for i in range(all_params):
#         alms_j = np.zeros_like(alms_input[i])
#         for j in range(all_params):
#             alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:,i,j], inplace=False)
#         alms_output[i] = np.copy(alms_j)
#     maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
#     if nstokes != 1:
#         return maps_output[3-nstokes:,...]
#     return maps_output
#     # return hp.alm2map(alms_output, nside, lmax=lmax)

