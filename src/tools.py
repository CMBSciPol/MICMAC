import os, sys, time
import numpy as np
import healpy as hp
import scipy
import scipy.linalg

def get_reduced_matrix_from_c_ell(c_ells_input):
    """ Expect c_ells_input to be sorted as TT, EE, BB, TE, TB, EB if 1, 3 or 6 spectra are given

        Generate covariance matrix from c_ells assuming it's block diagonal
    """
    c_ells_array = np.copy(c_ells_input)
    number_correlations = c_ells_array.shape[0]
    lmax = c_ells_array.shape[1]
    if number_correlations == 1:
        nstokes = 1
    elif number_correlations == 3:
        nstokes = 2
        # c_ells_array = np.vstack((c_ells_array, np.zeros(lmax)))
        number_correlations = 3
    elif number_correlations > 3:
        nstokes = 3
        if number_correlations != 6:
            for i in range(6 - number_correlations):
                c_ells_array = np.vstack((c_ells_array, np.zeros(lmax)))
            number_correlations = 6

    reduced_matrix = np.zeros((lmax,nstokes,nstokes))

    for i in range(nstokes):
        reduced_matrix[:,i,i] =  c_ells_array[i,:]
    
    # for j in range(number_correlations-nstokes):
    if number_correlations > 1:
        reduced_matrix[:,0,1] =  c_ells_array[nstokes,:]
        reduced_matrix[:,1,0] =  c_ells_array[nstokes,:]

    if number_correlations == 6:
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
    
    lmax = red_cov_mat.shape[0]
    nstokes = red_cov_mat.shape[1]

    number_correl = int(np.ceil(nstokes**2/2) + np.floor(nstokes/2))
    c_ells = np.zeros((number_correl, lmax))

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
    # if lmax == -1:
    # lmax = c_ells_input.shape[1]

    return np.linalg.pinv(get_reduced_matrix_from_c_ell(c_ells_input))[lmin:,...]

def get_inverse_reduced_matrix_from_red_matrix(red_matrix):
    # # if lmax == -1:
    # lmax = red_matrix.shape[0]

    # reduced_inverted_matrix = np.zeros_like(red_matrix)
    # for ell in range(lmin,lmax):
    #     reduced_inverted_matrix[ell] = np.linalg.pinv(red_matrix[ell])
    # return reduced_inverted_matrix
    return np.linalg.pinv(red_matrix)

def get_cholesky_reduced_matrix_from_matrix(red_matrix, lmin=0):
    """ Return L Cholesky decomposition
    """
    # if lmax == -1:
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    reduced_cholesky_decomposition = np.zeros_like(red_matrix)
    if lmin < 2:
        for ell in range(lmin, 2):
            try :
                reduced_cholesky_decomposition[ell,:,:] = np.linalg.cholesky(red_matrix[ell,:,:])
            except:
                reduced_cholesky_decomposition[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
                eigvals, eigevcs = np.linalg.eig(red_matrix[ell,:,:])
                # if not(np.allclose(eigvals, np.zeros(nstokes))):
                if np.any(eigvals<0) or not(np.allclose(eigvals, np.zeros(nstokes))):
                    print("Cannot get Cholesky decomposition of monopole of dipole :", ell, flush=True)

            
    for ell in range(max(2,lmin),lmax):
        # print("ell :", ell, "matrix :", red_matrix[ell,:,:])
        reduced_cholesky_decomposition[ell,:,:] = np.linalg.cholesky(red_matrix[ell,:,:])
    return reduced_cholesky_decomposition

def get_sqrt_reduced_matrix_from_matrix_old(red_matrix):
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
        reduced_sqrtm[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
    return reduced_sqrtm
    # eigvals, eigvect = np.linalg.eigh(red_matrix)
    # print("Test 1", eigvect, flush=True)
    # inv_eigvect = np.linalg.pinv(eigvect)
    # return np.einsum('ljk,km,lm,lmn->ljn', eigvect, np.eye(nstokes), np.sqrt(eigvals), inv_eigvect)
    # return np.einsum('ljk,km,lm,lmn->ljn', eigvect, np.eye(nstokes), np.sqrt(eigvals), np.linalg.pinv(eigvect))

def get_sqrt_reduced_matrix_from_matrix(red_matrix):
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

        reduced_sqrtm[ell] = np.einsum('jk,km,m,mn->jn', eigvect, np.eye(nstokes), np.sqrt(eigvals), inv_eigvect)
    return reduced_sqrtm

def get_sqrt_reduced_matrix_from_matrix_new(red_matrix):
    """ Return L square root matrix
    """
    # if lmax == -1:
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    # reduced_sqrtm = np.zeros_like(red_matrix)
    # # if np.any(np.iscomplex(red_matrix)):
    # #     print("COMPLEX ELEMENT IN RED MATRIX !!!", flush=True)
    # # for ell in range(lmin,lmax+lmin):
    # for ell in range(red_matrix.shape[0]):
    #     # if np.any(np.iscomplex(scipy.linalg.sqrtm(red_matrix[ell,:,:]))):
    #     #     print("COMPLEX ELEMENT IN SQRT RED MATRIX FOR ELL = {} !!!".format(ell), flush=True)
    #     reduced_sqrtm[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
    eigvals, eigvect = np.linalg.eigh(red_matrix)
    print("Test 1", eigvect, flush=True)
    inv_eigvect = np.linalg.pinv(eigvect)
    return np.einsum('ljk,km,lm,lmn->ljn', eigvect, np.eye(nstokes), np.sqrt(eigvals), inv_eigvect)
    return np.einsum('ljk,km,lm,lmn->ljn', eigvect, np.eye(nstokes), np.sqrt(eigvals), np.linalg.pinv(eigvect))

def get_cell_from_map(pixel_maps, lmax, n_iter=8):
    
    if len(pixel_maps.shape) == 1:
        nstokes = 1
    else:
        nstokes = pixel_maps.shape[0]
    
    if nstokes == 2:
            pixel_maps_for_Wishart = np.vstack((np.zeros_like(pixel_maps[0]), pixel_maps))
            print("Test 5 :", pixel_maps_for_Wishart.shape, pixel_maps_for_Wishart[0].mean(), pixel_maps_for_Wishart[1].mean(), pixel_maps_for_Wishart[2].mean())
    else:
        pixel_maps_for_Wishart = pixel_maps

    c_ells_Wishart = hp.anafast(pixel_maps_for_Wishart, lmax=lmax, iter=n_iter)

    if nstokes == 2:
        polar_indexes = np.array([1,2,4])
        # c_ells_Wishart = c_ells_Wishart[polar_indexes,self.lmin:]
        # print("Test 6a :", c_ells_Wishart.shape, c_ells_Wishart[0].mean(), c_ells_Wishart[1].mean(), c_ells_Wishart[2].mean(), c_ells_Wishart[3].mean(), c_ells_Wishart[4].mean())
        c_ells_Wishart = c_ells_Wishart[polar_indexes]
        # print("Test 6b :", c_ells_Wishart.shape, c_ells_Wishart[0].mean(), c_ells_Wishart[1].mean(), c_ells_Wishart[2].mean())
        
    return c_ells_Wishart

def generalized_cg_from_func(initial_guess, func_left_term, right_term, limit_iter_cg=100, tolerance=10**(-12)):
    """ Generalized CG from a function
        If it didn't converge, exit_code will be -1, otherwise 0
    """
    # print("Test PCG - 0", flush=True)
    new_residual = right_term - func_left_term(initial_guess)
    # print("Test PCG - 0b", flush=True)

    if np.linalg.norm(new_residual,ord=2) < tolerance * np.linalg.norm(right_term,ord=2):
        return new_residual
    
    vector_p = np.copy(new_residual)
    k = 0
    # previous_variable = np.copy(initial_guess)
    new_variable = np.copy(initial_guess)
    exit_code = -1
    
    while (k < limit_iter_cg) :
        previous_variable = np.copy(new_variable)
        residual = np.copy(new_residual)

        # print("Test PCG - 1", vector_p, flush=True)
        left_times_p = func_left_term(vector_p)
        # print("Test PCG - 1b", flush=True)

        alpha = np.dot(residual.T,residual)/np.dot(vector_p.T, left_times_p)

        new_variable = previous_variable + alpha*vector_p
        new_residual = residual - alpha * left_times_p
        # if np.linalg.norm(new_residual,ord=2) < tolerance * np.linalg.norm(right_term,ord=2):
        #     exit_code = 0
        #     break
        # if np.all(np.linalg.norm(new_residual.reshape(3,12*64**2),ord=2,axis=1) < tolerance * np.linalg.norm(right_term.reshape(3,12*64**2),ord=2,axis=1)):
        if np.linalg.norm(new_residual,ord=2) < tolerance * np.linalg.norm(right_term, ord=2):
            exit_code = 0
            break

        beta_k = np.dot(new_residual.T,new_residual)/np.dot(residual.T,residual)
        vector_p = new_residual + beta_k*vector_p
        k += 1

    return new_variable, k, exit_code

def maps_x_reduced_matrix_generalized(maps_TQU_input, red_matrix, lmin=0, n_iter=0):
    # if lmax == -1:
    lmax = red_matrix.shape[0]
    nstokes = red_matrix.shape[1]

    if len(maps_TQU_input.shape) == 1:
        nside = int(np.sqrt(len(maps_TQU_input)/12))
    else:
        nside = int(np.sqrt(len(maps_TQU_input[0])/12))
    
    # red_cholesky_decomp = np.zeros_like(red_matrix)
    red_cholesky_decomp = get_cholesky_reduced_matrix_from_matrix(red_matrix, lmin=lmin)
    # red_cholesky_decomp = get_sqrt_reduced_matrix_from_matrix(red_matrix, lmin=lmin)
    # if lmin < 2:
    #     for ell in range(lmin, 2):
    #         try :
    #             red_cholesky_decomp[ell,:,:] = np.linalg.cholesky(red_matrix[ell,:,:]).T
    #         except:
    #             red_cholesky_decomp[ell,:,:] = scipy.linalg.sqrtm(red_matrix[ell,:,:])
    #             eigvals, eigevcs = np.linalg.eig(red_matrix[ell,:,:])
    #             # if not(np.allclose(eigvals, np.zeros(nstokes))):
    #             if np.any(eigvals<0) or not(np.allclose(eigvals, np.zeros(nstokes))):
    #                 print("Can't get Cholesky decomposition of monopole of dipole :", ell, flush=True)


    # for ell in range(2,lmax):
    #     # print("ell", ell, red_matrix[ell,:,:])
    #     red_cholesky_decomp[ell,:,:] = np.linalg.cholesky(red_matrix[ell,:,:]).T

    if maps_TQU_input.shape[0] == 2:
        maps_TQU = np.vstack((np.zeros_like(maps_TQU_input[0]),maps_TQU_input))
    else:
        maps_TQU = np.copy(maps_TQU_input)

    alms_input = hp.map2alm(maps_TQU, lmax=lmax-1, iter=n_iter)
    # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    alms_output = np.zeros_like(alms_input)

    for i in range(nstokes):
        alms_j = np.zeros_like(alms_input[i])
        for j in range(nstokes):
            # alms_j += hp.almxfl(alms_input[j], red_cholesky_decomp[:,i,j], inplace=False)
            alms_j += hp.almxfl(alms_input[j], red_cholesky_decomp[:,j,i], inplace=False)
        alms_output[i] = np.copy(alms_j)
    return hp.alm2map(alms_output, nside, lmax=lmax-1)
    # return hp.alm2map(alms_output, nside, lmax=lmax)

def maps_x_reduced_matrix_generalized_sqrt_sqrt(maps_TQU_input, red_matrix_sqrt, lmin=0, n_iter=0):
    # if lmax == -1:
    lmax = red_matrix_sqrt.shape[0] - 1 + lmin
    nstokes = red_matrix_sqrt.shape[1]

    if len(maps_TQU_input.shape) == 1:
        nside = int(np.sqrt(len(maps_TQU_input)/12))
    else:
        nside = int(np.sqrt(len(maps_TQU_input[0])/12))
    
    # red_sqrt_decomp = np.zeros_like(red_matrix)
    red_sqrt_decomp = np.zeros((lmax+1,nstokes,nstokes))
    red_sqrt_decomp[lmin:,...] = red_matrix_sqrt

    if maps_TQU_input.shape[0] == 2:
        maps_TQU = np.vstack((np.zeros_like(maps_TQU_input[0]),maps_TQU_input))
    else:
        maps_TQU = np.copy(maps_TQU_input)

    alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    alms_output = np.zeros_like(alms_input)

    for i in range(nstokes):
        alms_j = np.zeros_like(alms_input[i])
        for j in range(nstokes):
            alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:,i,j], inplace=False)
        alms_output[i] = np.copy(alms_j)
    maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3-nstokes:,...]
    return maps_output

def maps_x_reduced_matrix_generalized_sqrt(maps_TQU_input, red_matrix, lmin=0, n_iter=0):
    # if lmax == -1:
    lmax = red_matrix.shape[0] - 1 + lmin
    nstokes = red_matrix.shape[1]

    if len(maps_TQU_input.shape) == 1:
        nside = int(np.sqrt(len(maps_TQU_input)/12))
    else:
        nside = int(np.sqrt(len(maps_TQU_input[0])/12))
    
    # red_sqrt_decomp = np.zeros_like(red_matrix)
    red_sqrt_decomp = np.zeros((lmax+1,nstokes,nstokes))
    red_sqrt_decomp[lmin:,...] = get_sqrt_reduced_matrix_from_matrix(red_matrix)

    if maps_TQU_input.shape[0] == 2:
        maps_TQU = np.vstack((np.zeros_like(maps_TQU_input[0]),maps_TQU_input))
    else:
        maps_TQU = np.copy(maps_TQU_input)

    alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    # alms_input = hp.map2alm(maps_TQU, lmax=lmax, iter=n_iter)
    alms_output = np.zeros_like(alms_input)

    for i in range(nstokes):
        alms_j = np.zeros_like(alms_input[i])
        for j in range(nstokes):
            alms_j += hp.almxfl(alms_input[j], red_sqrt_decomp[:,i,j], inplace=False)
        alms_output[i] = np.copy(alms_j)
    maps_output = hp.alm2map(alms_output, nside, lmax=lmax)
    if nstokes != 1:
        return maps_output[3-nstokes:,...]
    return maps_output
    # return hp.alm2map(alms_output, nside, lmax=lmax)

