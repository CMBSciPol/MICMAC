import healpy as hp
import jax.numpy as jnp
import jax.random as random
import numpy as np
import scipy as sp

import micmac


def test_red_covariance():
    """
    Testing the reduced covariance matrix routines
    """
    lmax = 128
    lmin = 2
    nstokes = 2

    n_correlations = int(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))

    fake_c_ell = jnp.arange(n_correlations * (lmax + 1 - lmin)).reshape((n_correlations, lmax + 1 - lmin))

    # First test the reduced covariance matrix creation routine
    reduced_matrix_from_routine = micmac.get_reduced_matrix_from_c_ell_jax(fake_c_ell)

    reduced_matrix_einsum = jnp.einsum('l,ij->lij', jnp.arange(lmin, lmax + 1), jnp.eye(nstokes))

    reduced_matrix_einsum = reduced_matrix_einsum.at[:, 0, 0].set(fake_c_ell[0, :])
    reduced_matrix_einsum = reduced_matrix_einsum.at[:, 1, 1].set(fake_c_ell[1, :])
    reduced_matrix_einsum = reduced_matrix_einsum.at[:, 1, 0].set(fake_c_ell[nstokes, :])
    reduced_matrix_einsum = reduced_matrix_einsum.at[:, 0, 1].set(fake_c_ell[nstokes, :])

    assert jnp.allclose(reduced_matrix_from_routine, reduced_matrix_einsum)

    # Now test the c_ell retrieval routine from reduced covariance matrix

    c_ell_retrieved = micmac.get_c_ells_from_red_covariance_matrix_JAX(reduced_matrix_from_routine)

    assert jnp.allclose(c_ell_retrieved, fake_c_ell)


def test_sqrt():
    """
    Testing the sqrt of a matrix
    """
    lmax = 128
    lmin = 2
    nstokes = 2

    n_correlations = int(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))

    fake_c_ell = jnp.flip(jnp.arange(n_correlations * (lmax + 1 - lmin))).reshape((n_correlations, lmax + 1 - 2))

    reduced_matrix_from_routine = micmac.get_reduced_matrix_from_c_ell_jax(fake_c_ell)

    # Get matrix sqrt from routine
    red_matrix_sqrt = micmac.get_sqrt_reduced_matrix_from_matrix_jax(reduced_matrix_from_routine)

    for i in range(lmax + 1 - lmin):
        element_sqrt = jnp.float64(sp.linalg.sqrtm(reduced_matrix_from_routine[i, ...]))
        assert jnp.allclose(element_sqrt, red_matrix_sqrt[i, ...])


def test_c_ell_from_map():
    nside = 64
    lmin = 2
    nstokes = 2

    n_iter = 8

    lmax = 2 * nside
    n_correlations = int(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))
    n_pix = 12 * nside**2

    index_polar = jnp.array([1, 2, 4])

    fake_map = random.normal(random.PRNGKey(0), shape=(nstokes, n_pix))

    c_ell_micmac = micmac.get_cell_from_map_jax(fake_map, lmax, n_iter=n_iter)

    fake_map_extended = np.vstack([jnp.zeros_like(fake_map[0]), fake_map])
    c_ell_hp = hp.anafast(fake_map_extended, lmax=lmax, iter=n_iter)[index_polar]

    assert jnp.all(jnp.abs(c_ell_micmac - c_ell_hp) < 10**-8)


def test_alms():
    """
    Test routines around alms
    """

    nside = 64
    lmin = 2
    nstokes = 2

    n_pix = 12 * nside**2

    lmax = 2 * nside
    n_correlations = int(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))

    fake_c_ells = np.einsum('l,i->il', np.arange(lmax + 1), np.ones(4))
    fake_c_ells[:, :lmin] = 0
    fake_c_ells[0, :] = 0
    fake_c_ells[-1, :] = 0

    fake_map = random.normal(random.PRNGKey(0), shape=(nstokes, n_pix))
    fake_map_extended = np.vstack([jnp.zeros_like(fake_map[0]), fake_map])

    alms_1 = hp.synalm(fake_c_ells, lmax=lmax, new=True)
    alms_2 = hp.map2alm(fake_map_extended, lmax=lmax, pol=True)

    # First test on alm x alm
    alm_1_x_alm_2_micmac = micmac.alm_dot_product_JAX(alms_1, alms_2, lmax)

    alm_1_x_alm_2_hp = (
        jnp.array(hp.alm2cl(alms_1, alms_2, lmax=lmax))[1 : nstokes + 1] * (2 * jnp.arange(lmax + 1) + 1)
    ).sum()

    assert jnp.all(jnp.abs(alm_1_x_alm_2_micmac - alm_1_x_alm_2_hp) < 1e-10)

    # Second test on alm x fl

    alm_x_c_ell_micmac = micmac.JAX_almxfl(alms_1[1], fake_c_ells[1], lmax)

    alm_x_c_ell_hp = jnp.array(hp.almxfl(alms_1[1], fake_c_ells[1]))

    assert jnp.all(jnp.abs(alm_x_c_ell_micmac - alm_x_c_ell_hp) < 1e-10)

    # TODO: alms_x_red_covariance_cell_JAX ; frequency_alms_x_obj_red_covariance_cell_JAX


def test_maps_x_red_cov():
    nside = 64
    lmin = 2
    nstokes = 2
    n_iter = 8

    n_pix = 12 * nside**2

    lmax = 2 * nside
    n_correlations = int(jnp.ceil(nstokes**2 / 2) + jnp.floor(nstokes / 2))

    indices_polar = np.array([1, 2, 4])

    # First test with diagonal c_ells

    fake_c_ells = 10 * np.einsum('l,i->il', np.zeros(lmax + 1), np.zeros(3))

    fake_c_ells[:nstokes, lmin:] = 1
    fake_c_ell_extended = np.zeros((6, lmax + 1))
    fake_c_ell_extended[indices_polar] = fake_c_ells

    fake_alms = hp.synalm(fake_c_ell_extended, lmax=lmax, new=True)
    fake_map = hp.alm2map(fake_alms, nside=nside, pol=True)[1:]

    fake_red_c_ells = micmac.get_reduced_matrix_from_c_ell_jax(fake_c_ells)
    final_test_map = micmac.maps_x_red_covariance_cell_JAX(
        fake_map, 2 * fake_red_c_ells, nside=nside, lmin=0, n_iter=n_iter
    )

    assert jnp.all(jnp.abs(final_test_map - 2 * fake_map) < 1e-6)
