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


def get_MCMC_batch_error(sample_single_chain, batch_size):
    # number_iterations = np.size(sample_single_chain, axis=0)
    number_iterations = sample_single_chain.shape[0]
    assert number_iterations % batch_size == 0

    overall_mean = np.average(sample_single_chain, axis=0)
    standard_error = np.sqrt((batch_size / number_iterations) * ((sample_single_chain - overall_mean) ** 2).sum())
    return standard_error


def get_empirical_covariance_JAX(samples):
    """Compute empirical covariance from samples"""
    number_samples = jnp.size(samples, axis=0)
    mean_samples = jnp.mean(samples, axis=0)

    return (
        jnp.einsum('ti,tj->tij', samples, samples).sum(axis=0)
        - number_samples * jnp.einsum('i,j->ij', mean_samples, mean_samples)
    ) / (number_samples - 1)


def get_Gelman_Rubin_statistics(all_chain_samples):
    """Compute Gelman-Rubin statistics

    Parameters
    ----------
    :param all_chains_samples: all chains, with dimensions [n_chains, number_iterations, ...]
    """

    mean_chain = all_chain_samples.mean(axis=0)

    return 1 / all_chain_samples.var(axis=1).mean() * mean_chain.var(axis=0)
