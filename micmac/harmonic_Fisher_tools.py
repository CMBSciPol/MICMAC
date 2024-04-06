import healpy as hp
import numpy as np

OPTIMIZE = 'optimal'


def _mv(m, v):
    return np.einsum('...ij,...j->...i', m, v, optimize=OPTIMIZE)


# Added by Clement Leloup
def _uvt(u, v):
    return np.einsum('...i,...j->...ij', u, v)


def _mm(m, n):
    return np.einsum('...ij,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mtm(m, n):
    return np.einsum('...ji,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mtmv(m, w, v):
    return np.einsum('...ji,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mmm(m, w, n):
    return np.einsum('...ij,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)


def _mtmm(m, w, n):
    return np.einsum('...ji,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)


def _format_prior(c_ells, lmax):
    ell = hp.Alm.getlm(lmax)[0]
    ell = np.stack((ell, ell), axis=-1).reshape(-1)

    c_ellm = np.array([c_ells[l, :] for l in ell])  # [:,np.newaxis,:]
    c_ellm[np.arange(1, 2 * (lmax + 1), 2), ...] = 0  # to avoid overcounting for m=0
    # c_ellm = np.swapaxes(c_ellm, 0, -1)
    return c_ellm


def _format_alms(alms_c, lmin=0):
    alms = alms_c * 1.0  # Wang: extra line added
    lmax = hp.Alm.getlmax(alms.shape[-1])
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    em = hp.Alm.getlm(lmax)[1]
    em = np.stack((em, em), axis=-1).reshape(-1)
    # alms *= (-1)**em   # Should be there in principle but when no oupling between m's, alms only appear squared in the likelihood
    mask_em = [m != 0 for m in em]
    alms[..., mask_em] *= np.sqrt(2)
    alms[..., np.arange(1, 2 * (lmax + 1), 2)] = hp.UNSEEN  # Mask imaginary m = 0
    # mask_alms = _intersect_mask(alms)
    # alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood
    alms = np.swapaxes(alms, 0, -1)

    if lmin != 0:
        ell = hp.Alm.getlm(lmax)[0]
        ell = np.stack((ell, ell), axis=-1).reshape(-1)
        mask_lmin = [l < lmin for l in ell]
        alms[mask_lmin, ...] = 0

    return alms


def _get_alms(data_map, beams=None, lmax=None, weights=None, iter=3):
    alms = []
    for f, fdata in enumerate(data_map):
        if weights is None:
            alms.append(hp.map2alm(fdata, lmax=lmax, iter=iter))
        else:
            alms.append(hp.map2alm(hp.ma(fdata) * weights, lmax=lmax, iter=iter))
            # logging.info(f"{f+1} of {len(data)} complete")
        print(f'{f+1} of {len(data_map)} complete', flush=True)
    alms = np.array(alms)

    if beams is not None:
        # logging.info('Correcting alms for the beams')
        print('Correcting alms for the beams', flush=True)
        for fwhm, alm in zip(beams, alms):
            bl = hp.gauss_beam(np.radians(fwhm / 60.0), lmax, pol=(alm.ndim == 2))

            if alm.ndim == 1:
                alm = [alm]
                bl = [bl]

            for i_alm, i_bl in zip(alm, bl.T):
                hp.almxfl(i_alm, 1.0 / i_bl, inplace=True)

    return alms


def nonparam_biascorr_fisher_bruteforce(A, s, Sc, Sc_approx, A_dB, Sc_dr, invN=None):
    Fisher = np.zeros((len(A_dB) + 1, len(A_dB) + 1))

    try:
        cov = np.linalg.inv(_mtmm(A, invN, A))
    except np.linalg.LinAlgError:
        cov = np.zeros_like(_mtmm(A, invN, A))
        good_idx = np.where(np.all(np.diagonal(_mtmm(A, invN, A), axis1=-1, axis2=-2), axis=-1))
        cov[good_idx] = np.linalg.inv(_mtmm(A, invN[good_idx], A))

    Nc = cov[..., 0, 0]
    P = invN - _mmm(_mm(invN, A), cov, _mtm(A, invN))

    # Define CMB average of s*T(s)
    sst_avg = np.zeros((Sc.shape[0], s.shape[1], s.shape[-1], s.shape[-1]))
    sst_avg[..., 0, 0] = Sc
    sst_avg[..., 1:, 1:] = _uvt(s[..., 1:], s[..., 1:])

    for i in np.arange(len(A_dB)):
        A_dB_i = A_dB[i]
        print('A_dB = ', A_dB_i)
        for j in np.arange(len(A_dB)):
            A_dB_j = A_dB[j]

            Nc_dB = -2.0 * _mmm(cov, _mtmm(A_dB_i, invN, A), cov)[..., 0, 0]
            Nc_dB_prime = -2.0 * _mmm(cov, _mtmm(A_dB_j, invN, A), cov)[..., 0, 0]

            with np.errstate(divide='ignore'):
                m1_dBdB = (
                    2.0 * np.trace(_mtmm(A_dB_i, P, _mm(A_dB_j, sst_avg)), axis1=-1, axis2=-2)
                    - Nc_dB * Nc_dB_prime / ((Sc + Nc) ** 2)
                    + Nc_dB * Nc_dB_prime / ((Sc_approx + Nc) ** 2)
                )
                m2_dBdB = (
                    -2.0
                    * (Sc - Sc_approx)
                    / ((Sc_approx + Nc) * (Sc + Nc))
                    * _mmm(
                        cov,
                        _mmm(_mtmm(A, invN, A_dB_i), cov, _mtmm(A, invN, A_dB_j))
                        + _mmm(_mtmm(A, invN, A_dB_j), cov, _mtmm(A, invN, A_dB_i))
                        + _mmm(_mtmm(A, invN, A_dB_i), cov, _mtmm(A_dB_j, invN, A))
                        - _mtmm(A_dB_i, P, A_dB_j),
                        cov,
                    )[..., 0, 0]
                )
                m3_dBdB = (
                    2.0
                    / (Sc + Nc)
                    * _mmm(cov, _mmm(_mtmm(A, invN, A_dB_i), sst_avg, _mtmm(A_dB_j, invN, A)), cov)[..., 0, 0]
                )
            m1_dBdB[~np.isfinite(m1_dBdB)] = 0.0
            m2_dBdB[~np.isfinite(m2_dBdB)] = 0.0
            m3_dBdB[~np.isfinite(m3_dBdB)] = 0.0
            Fisher[i, j] = np.sum(m1_dBdB + m2_dBdB + m3_dBdB)

    m_drdr = np.zeros(Sc.shape)
    with np.errstate(divide='ignore'):
        m_drdr = (Sc_dr / (Sc + Nc)) ** 2
    m_drdr[~np.isfinite(m_drdr)] = 0.0
    Fisher[-1, -1] = np.sum(m_drdr)
    print(Fisher)

    w, v = np.linalg.eig(Fisher)
    print('eigenvalues : ', w)
    cond = np.linalg.cond(Fisher)
    print('condition number : ', cond)

    return Fisher


def full_fisher(
    init_mixing_matrix_obj,
    exact_params_mixing_matrix,
    Cl_lens,
    Cl_prim,
    data_map,
    lmax,
    inv_Nl,
    lmin=2,
    r_start=0.001,
    mode='EB',
    Sc_approx=None,
):
    # Format objects
    ell = hp.Alm.getlm(lmax)[0]
    ell = np.stack((ell, ell), axis=-1).reshape(-1)
    mask_lmin = [l >= lmin for l in ell]

    alms = _get_alms(data_map, lmax=lmax)[:, 1:, :]
    alms = _format_alms(alms, lmin=0)
    alms = alms[mask_lmin, ...]

    invNlm = np.array([inv_Nl.T[l, :, :] for l in ell])[:, np.newaxis, :, :]
    invNlm[np.arange(1, 2 * (lmax + 1), 2), ...] = 0  # to avoid overcounting for m=0
    invNlm = invNlm[mask_lmin, ...]

    assert mode in ['B', 'E', 'EB']

    print('Computing mixing matrix')
    B_matrix = init_mixing_matrix_obj.get_B().mean(axis=2)
    B_dB_matrix = init_mixing_matrix_obj.get_B_db()

    x0 = np.append(np.ravel(exact_params_mixing_matrix, order='F'), r_start)  # true values of the parameters

    # Format C and C_approx
    ell_in = np.arange(2, lmax + 1)
    mask_lmin = [l >= lmin for l in ell]

    if mode == 'B':  # B modes only
        Cl_prim = _format_prior(Cl_prim[2:3, : lmax + 1].T, lmax)
        Cl_lens = _format_prior(Cl_lens[2:3, : lmax + 1].T, lmax)
        if Sc_approx is not None:
            Sc_approx = _format_prior(Sc_approx[2:3, : lmax + 1].T, lmax)
    elif mode == 'E':  # E modes only
        Cl_prim = _format_prior(Cl_prim[1:2, : lmax + 1].T, lmax)
        Cl_lens = _format_prior(Cl_lens[1:2, : lmax + 1].T, lmax)
        if Sc_approx is not None:
            Sc_approx = _format_prior(Sc_approx[1:2, : lmax + 1].T, lmax)
    else:  # E and B modes
        Cl_prim = _format_prior(Cl_prim[1:3, : lmax + 1].T, lmax)
        Cl_lens = _format_prior(Cl_lens[1:3, : lmax + 1].T, lmax)
        if Sc_approx is not None:
            Sc_approx = _format_prior(Sc_approx[1:3, : lmax + 1].T, lmax)

    Cl_prim = Cl_prim[mask_lmin, ...]
    Cl_lens = Cl_lens[mask_lmin, ...]
    if Sc_approx is not None:
        Sc_approx = Sc_approx[mask_lmin, ...]

    # Start evaluating Fisher
    try:
        cov = np.linalg.inv(_mtmm(B_matrix, invNlm, B_matrix))
    # except np.linalg.LinAlgError:
    except:
        cov = np.zeros_like(_mtmm(B_matrix, invNlm, B_matrix))
        good_idx = np.where(np.all(np.diagonal(_mtmm(B_matrix, invNlm, B_matrix), axis1=-1, axis2=-2), axis=-1))
        cov[good_idx] = np.linalg.inv(_mtmm(B_matrix, invNlm[good_idx], B_matrix))

    s = _mv(cov, _mtmv(B_matrix, invNlm, alms))  # True s

    Fisher = 0.5 * nonparam_biascorr_fisher_bruteforce(
        B_matrix, s, x0[-1] * Cl_prim + Cl_lens, Sc_approx, B_dB_matrix, Cl_prim, invNlm
    )
    w, v = np.linalg.eig(Fisher)
    print('eigenvalues : ', w)
    cond = np.linalg.cond(Fisher)
    print('condition number : ', cond)

    return Fisher
