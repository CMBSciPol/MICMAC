import os, sys, time
import numpy as np
import scipy.linalg

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
