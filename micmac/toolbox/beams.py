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

__all__ = ['get_beam_harmonic']


def get_beam_harmonic(ell_range, sigma_fwhm, spin=2):
    """
    Compute the harmonic-space beam response for a given range of multipoles.
    sigma_fwhm is the full-width at half-maximum of the Gaussian beam in radians.
    """
    sigma = sigma_fwhm / jnp.sqrt(8 * jnp.log(2))
    if jnp.array(sigma).size > 1:
        return jnp.exp(jnp.einsum('l,f->fl', -0.5 * (ell_range * (ell_range + 1) - spin**2), sigma**2))
    return jnp.exp(-0.5 * (ell_range * (ell_range + 1) - spin**2) * sigma**2)
