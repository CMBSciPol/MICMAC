import numpy as np
import scipy as sp
import healpy as hp


def get_log_gaussian(variable, inverse_covariance):
    """ Compute log of the Gaussian without renormalization factor which doesn't depend on variable or inverse_covariance
    """
    return np.linalg.multi_dot((-variable.T, inverse_covariance, variable))/2 + np.log(np.linalg.det(inverse_covariance))/2


def get_multivariate_gamma(variable, dim):
    """ Compute multivariate gamma function
    """
    return np.pi**(dim*(dim-1)/4)*np.prod(sp.special.gamma(variable + 1 - (np.arange(dim)+1)/2))

def get_proba_inverse_wishart_distribution(variable, param_matrix, ddl):
    """ Compute the value of the inverse Wishart distribution probability for a variable, and given a matrix parameter and degrees of freedom
        The variable and parameter matrix must be positive definite
    """
    assert len(variable.shape) == 2
    assert np.all(np.linalg.eigh(param_matrix)[0] > 0)
    assert np.all(np.linalg.eigh(variable)[0] > 0)
    dim = variable.shape[0]
    assert ddl > 2*dim

    exposant = ddl - dim - 1
    coefficient = np.linalg.det(param_matrix)**(.5*exposant)/(2**(.5*exposant*dim) * get_multivariate_gamma(.5*exposant, dim) * np.linalg.det(variable)**(.5*ddl))
    return coefficient*np.exp(-.5*np.einsum('ij,ji', np.linalg.pinv(variable), param_matrix))

def get_generalized_proba_inverse_wishart_distribution(variable, param_matrix, ddl):
    """ Compute the value of the inverse Wishart distribution probability for a variable, and given a matrix parameter and degrees of freedom
        The parameter matrix must be positive definite
    """
    assert len(variable.shape) == 3
    assert np.all(np.linalg.eigh(param_matrix)[0] > 0)
    # assert np.all(np.linalg.eigh(variable)[0] > 0)
    dim = variable.shape[1]
    # assert ddl > 2*dim

    exposant = ddl - dim - 1
    coefficient = np.linalg.det(param_matrix)**(.5*exposant)/(2**(.5*exposant*dim) * get_multivariate_gamma(.5*exposant, dim) * np.linalg.det(variable)**(.5*ddl))
    return coefficient*np.exp(-.5*np.einsum('kij,ji', np.linalg.pinv(variable), param_matrix))


def get_proba_inverse_gamma_distribution_ordinary(variable, param, ddl):
    """ Compute the inverse gamma probability of a variable
    """
    return param**ddl/sp.special.gamma(ddl) * (1/variable)**(ddl+1) * np.exp(-param/variable)

def get_proba_inverse_gamma_distribution(variable, param, ddl, order=1):
    """ Compute the inverse gamma probability of a variable
    """
    assert np.all(variable > 0)
    assert ddl > 2*order
    return param**(.5*(ddl-2*order))/(2**(.5*(ddl-2*order))*sp.special.gamma(.5*(ddl-2*order))) * variable**(-.5*(ddl-2*order+2)) * np.exp(-.5*param/variable)

def get_log_proba_inverse_gamma_distribution(variable, param, ddl, order=1):
    """ Compute the logarithm of an inverse gamma probability of a variable, which is better to deal with very low quantities
    """
    assert np.all(variable > 0)
    assert ddl > 2*order
    return (.5*(ddl-2*order))*np.log(param) - np.log((2**(.5*(ddl-2*order))) + np.log(sp.special.gamma(.5*(ddl-2*order)))) + np.log(variable)*(-.5*(ddl-2*order+2)) -.5*param/variable
