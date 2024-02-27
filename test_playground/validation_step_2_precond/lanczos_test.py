import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

def lanczos_algorithm_JAX(matvec_func, right_member, initial_guess=None, max_iter=50):
    """ Perform Lanczos algorithm to obtain eigenvectors

        Parameters
        ----------
        :param matvec_func: function that takes a vector and returns the product of the matrix and the vector
        :param right_member: right member of the linear system of equations
        :param initial_guess: initial guess for the solution
        :param treshold: treshold for the convergence of the algorithm
        :param max_iter: maximum number of iterations

        Returns
        -------
        :return eigenvectors: eigenvectors of the matrix
    """
    if initial_guess is None:
        initial_guess = jnp.zeros_like(right_member)

    residual = right_member - matvec_func(initial_guess)

    first_beta = jnp.linalg.norm(residual, ord=2)
    
    first_eigenvector = residual/first_beta
    first_matvec_to_store = matvec_func(first_eigenvector)
    first_alpha = jnp.dot(first_eigenvector, first_matvec_to_store)

    def perform_1Lanczos_iteration(carry, iteration):
        alpha, beta, eigvec_k, eigvec_k_m1 = carry
        
        matvec_to_store = matvec_func(eigvec_k)
        
        w_b_k = matvec_to_store - beta*eigvec_k_m1

        alpha = jnp.dot(w_b_k, eigvec_k)

        w_k = w_b_k - alpha*eigvec_k
        
        beta = jnp.linalg.norm(w_k, ord=2)
        
        eigvec_k_p1 = w_k/beta
        
        return (alpha, beta, eigvec_k_p1, eigvec_k), (matvec_to_store, eigvec_k_p1, alpha, beta)
    
    initial_carry = (first_alpha, first_beta, first_eigenvector, jnp.zeros_like(first_eigenvector))
    initial_state = (first_matvec_to_store, first_eigenvector, first_alpha, first_beta)

    _, (matvec_to_store_, eigenvectors_, alpha_, beta_) = jax.lax.scan(perform_1Lanczos_iteration, initial_carry, jnp.arange(max_iter))

    all_alphas = jnp.concatenate((jnp.array([first_alpha]), alpha_))
    all_betas = jnp.concatenate((jnp.array([first_beta]), beta_))
    all_eigenvectors = jnp.vstack((jnp.expand_dims(first_eigenvector, axis=0), eigenvectors_))
    all_matvec_to_store = jnp.vstack((jnp.expand_dims(first_matvec_to_store, axis=0), matvec_to_store_))

    return all_alphas, all_betas, all_eigenvectors, all_matvec_to_store


def construct_partial_2lvl_preconditioners_JAX(alphas, betas, eigenvectors, matvec_stored):
    """ Construct 2-level preconditioner from Lanczos algorithm computations

        Parameters
        ----------
        :param alphas: alphas from Lanczos algorithm
        :param betas: betas from Lanczos algorithm
        :param eigenvectors: eigenvectors from Lanczos algorithm
        :param matvec_stored: matrix-vector products from Lanczos algorithm

        Returns
        -------
        :return preconditioner_matrix: preconditioner matrix
    """

    # eigvals_tridiag, eigvects_tridiag = jsp.linalg.eigh_tridiagonal(alphas, betas[1:], select='a', eigvals_only=False)
    tridiag_matri = jnp.diag(alphas) + jnp.diag(betas[1:], k=1) + jnp.diag(betas[1:], k=-1)
    eigvals_tridiag, eigvects_tridiag = jnp.linalg.eigh(tridiag_matri)

    

    deflation_matrix = jnp.einsum('ij,jk->ik', eigenvectors, eigvects_tridiag)
    matvec_x_deflation_matrix = jnp.einsum('ij,jk->ik', matvec_stored, eigvects_tridiag)

    partial_precond = jnp.einsum('jk,lk->jl',
                                 jnp.linalg.pinv(jnp.einsum('ba,bc',
                                                            deflation_matrix,
                                                            matvec_x_deflation_matrix)),
                                 deflation_matrix)

    first_term_sp_BCOO = jsparse.BCOO.fromdense(jnp.einsum('ij,jk->ik', matvec_stored, partial_precond))
    second_term_sp_BCOO = jsparse.BCOO.fromdense(jnp.einsum('ij,jk->ik', deflation_matrix, partial_precond))
    return first_term_sp_BCOO, second_term_sp_BCOO

def apply_preconditioner_JAX(partial_2lvl_precond, block_diag_precond_func, vector):
    """ Apply preconditioner to a vector

        Parameters
        ----------
        :param partial_2lvl_precond: 2-level preconditioner
        :param block_diag_precond_func: function that applies block-diagonal preconditioner
        :param vector: vector to which apply the preconditioner

        Returns
        -------
        :return preconditioned_vector: preconditioned vector
    """

    first_term = partial_2lvl_precond[0]@vector
    second_term = partial_2lvl_precond[1]@vector
    return block_diag_precond_func(vector-first_term) + second_term
