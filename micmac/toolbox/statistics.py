# This file is part of MICMAC.
# Copyright (C) 2024 CNRS / SciPol developers
#
# MICMAC is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MICMAC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MICMAC. If not, see <https://www.gnu.org/licenses/>.

import jax.numpy as jnp
import numpy as np

__all__ = ['get_Gelman_Rubin_statistics', 'get_1d_recursive_empirical_covariance']


def get_MCMC_batch_error(sample_single_chain, batch_size):
    """
    Not used
    """
    # number_iterations = np.size(sample_single_chain, axis=0)
    number_iterations = sample_single_chain.shape[0]
    assert number_iterations % batch_size == 0

    overall_mean = np.average(sample_single_chain, axis=0)
    standard_error = np.sqrt((batch_size / number_iterations) * ((sample_single_chain - overall_mean) ** 2).sum())
    return standard_error


def get_empirical_covariance_JAX(samples):
    """
    Not used

    Compute empirical covariance from samples
    """
    number_samples = jnp.size(samples, axis=0)
    mean_samples = jnp.mean(samples, axis=0)

    return (
        jnp.einsum('ti,tj->tij', samples, samples).sum(axis=0)
        - number_samples * jnp.einsum('i,j->ij', mean_samples, mean_samples)
    ) / (number_samples - 1)


def get_1d_recursive_empirical_covariance(
    iteration_number, last_sample, last_mean_samples, last_empirical_covariance, s_param=(2.4) ** 2, epsilon_param=1e-10
):
    """
    Compute the 1D empirical covariance recursively, from last sample computed and mean samples

    Parameters
    ----------
    iteration_number: int
        number of iterations corresponding to the last sample
    last_sample: float or array
        last sample computed
    last_mean_samples: float or array
        mean samples computed from the previous iteration_number samples
    last_empirical_covariance: float or array
        empirical covariance computed from the previous iteration_number samples

    Returns
    -------
    empirical_covariance: array
        empirical covariance computed from the iteration_number+1 samples
    """
    new_mean_samples = (iteration_number * last_mean_samples + last_sample) / (iteration_number + 1)
    # return (iteration_number - 1) / iteration_number * last_empirical_covariance + last_mean_samples ** 2 + last_sample ** 2 / iteration_number - (iteration_number + 1) * new_mean_samples ** 2 / iteration_number

    return (
        (iteration_number - 1) / iteration_number * last_empirical_covariance
        + s_param
        * (
            last_mean_samples**2
            + last_sample**2 / iteration_number
            - (iteration_number + 1) * new_mean_samples**2 / iteration_number
        )
        + s_param * epsilon_param
    )

    # return s_param * ((iteration_number - 1) / iteration_number * last_empirical_covariance + last_mean_samples ** 2 + last_sample ** 2 / iteration_number - (iteration_number + 1) * new_mean_samples ** 2 / iteration_number) + s_param * epsilon_param


def get_Gelman_Rubin_statistics(all_chain_samples):
    """
    Compute Gelman-Rubin statistics

    Parameters
    ----------
    all_chains_samples: array with dimensions [n_chains, number_iterations, ...]
        all chains

    Returns
    -------
    GR: float
        Gelman-Rubin statistics
    """

    mean_chain = all_chain_samples.mean(axis=0)

    return 1 / all_chain_samples.var(axis=1).mean() * mean_chain.var(axis=0)
