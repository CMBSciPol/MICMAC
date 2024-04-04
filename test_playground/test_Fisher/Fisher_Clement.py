import numpy as np
import healpy as hp


# Format objects                                                                                                                                                                                            
ell = hp.Alm.getlm(lmax)[0]
ell = np.stack((ell, ell), axis=-1).reshape(-1)
mask_lmin = [l >= lmin for l in ell]
alms = _format_alms(alms)
alms = alms[mask_lmin, ...]
invNlm = np.array([inv_Nl[l,:,:] for l in ell])[:,np.newaxis,:,:]
invNlm[np.arange(1, 2*(lmax+1), 2), ...] = 0 # to avoid overcounting for m=0                                                                                                                                
invNlm = invNlm[mask_lmin, ...]



OPTIMIZE = 'optimal'
def _inv(m):
    result = np.array(map(np.linalg.inv, m.reshape((-1,)+m.shape[-2:])))
    return result.reshape(m.shape)


def _mv(m, v):
    return np.einsum('...ij,...j->...i', m, v, optimize=OPTIMIZE)

#Added by Clement Leloup                                                                                                                                                                                    
def _utv(u, v):
    return np.einsum('...i,...i', u, v)
def _uvt(u, v):
    return np.einsum('...i,...j->...ij', u, v)
def _utm(u, m):
    return np.einsum('...i,...ij->...j', u, m, optimize=OPTIMIZE)

def _utmv(u, m, v):
    return np.einsum('...i,...ij,...j', u, m, v, optimize=OPTIMIZE)

def _mtv(m, v):
    return np.einsum('...ji,...j->...i', m, v, optimize=OPTIMIZE)

def _mm(m, n):
    return np.einsum('...ij,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mtm(m, n):
    return np.einsum('...ji,...jk->...ik', m, n, optimize=OPTIMIZE)


def _mmv(m, w, v):
    return np.einsum('...ij,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mtmv(m, w, v):
    return np.einsum('...ji,...jk,...k->...i', m, w, v, optimize=OPTIMIZE)


def _mmm(m, w, n):
    return np.einsum('...ij,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)


def _mtmm(m, w, n):
    return np.einsum('...ji,...jk,...kh->...ih', m, w, n, optimize=OPTIMIZE)

def _format_alms(alms_c, lmin=0):
    
    alms = alms_c * 1. # Wang: extra line added                                                                                                                                                             
    lmax = hp.Alm.getlmax(alms.shape[-1])
    alms = np.asarray(alms, order='C')
    alms = alms.view(np.float64)
    em = hp.Alm.getlm(lmax)[1]
    em = np.stack((em, em), axis=-1).reshape(-1)
    #alms *= (-1)**em   # Should be there in principle but when no oupling between m's, alms only appear squared in the likelihood                                                                          
    mask_em = [m != 0 for m in em]
    alms[..., mask_em] *= np.sqrt(2)
    alms[..., np.arange(1, 2*(lmax+1), 2)] = hp.UNSEEN  # Mask imaginary m = 0                                                                                                                              
    mask_alms = _intersect_mask(alms)
    alms[..., mask_alms] = 0  # Thus no contribution to the spectral likelihood                                                                                                                             
    alms = np.swapaxes(alms, 0, -1)

    if lmin != 0:
        ell = hp.Alm.getlm(lmax)[0]
        ell = np.stack((ell, ell), axis=-1).reshape(-1)
        mask_lmin = [l < lmin for l in ell]
        alms[mask_lmin, ...] = 0

    return alms

def _get_alms(data, beams=None, lmax=None, weights=None, iter=3):
    alms = []
    for f, fdata in enumerate(data):
	if weights is None:
            alms.append(hp.map2alm(fdata, lmax=lmax, iter=iter))
	else:
            alms.append(hp.map2alm(hp.ma(fdata)*weights, lmax=lmax, iter=iter))
        logging.info(f"{f+1} of {len(data)} complete")
    alms = np.array(alms)

    if beams is not None:
        logging.info('Correcting alms for the beams')
        for fwhm, alm in zip(beams, alms):
            bl = hp.gauss_beam(np.radians(fwhm/60.0), lmax, pol=(alm.ndim==2))

            if alm.ndim == 1:
		alm = [alm]
		bl = [bl]

            for i_alm, i_bl in zip(alm, bl.T):
                hp.almxfl(i_alm, 1.0/i_bl, inplace=True)

    return alms

def full_fisher(components, instrument, Cl_lens, Cl_prim, data, lmax, ell, invNlm, lmin=2, r_start=0.001, nblind=None, mode='B', Sc_approx=None):
    
    # nblind : number of non-param components in the mixing matrix                                                                                                                                          

    assert mode in ['B', 'E', 'EB']

    instrument = standardize_instrument(instrument) #useless
    if nblind is None:
	nblind = len(components)-1
    nfreq = np.array([0, -1]) #useless

    print('Computing mixing matrix')
    A_ev, A_dB_ev, comp_of_param, x0, params = init_nonparam_mixmat(components, instrument, nblind, nfreq=nfreq) #useless, to be replaced by MICMAC mixing matrix

    if not len(x0):
        print("No spectral parameter to maximize !")

    x0 = np.append(x0, r_start) # true values of the parameters                                                                                                                                             


    # Format C and C_approx                                                                                                                                                                                 
    ell_in = np.arange(2, lmax+1)
    mask_lmin = [l >= lmin for l in ell]

    if mode == 'B': # B modes only                                                                                                                                                                          
        Cl_prim = _format_prior(Cl_prim[2:3,:lmax+1].T, lmax)
        Cl_lens = _format_prior(Cl_lens[2:3,:lmax+1].T, lmax)
        if Sc_approx is not None:
            Sc_approx = _format_prior(Sc_approx[2:3,:lmax+1].T, lmax)
    elif mode == 'E': # E modes only                                                                                                                                                                        
        Cl_prim = _format_prior(Cl_prim[1:2,:lmax+1].T, lmax)
        Cl_lens = _format_prior(Cl_lens[1:2,:lmax+1].T, lmax)
        if Sc_approx is not None:
            Sc_approx = _format_prior(Sc_approx[1:2,:lmax+1].T, lmax)
    else: # E and B modes                                                                                                                                                                                   
        Cl_prim = _format_prior(Cl_prim[1:3,:lmax+1].T, lmax)
	Cl_lens = _format_prior(Cl_lens[1:3,:lmax+1].T, lmax)
	if Sc_approx is not None:
            Sc_approx = _format_prior(Sc_approx[1:3,:lmax+1].T, lmax)

    Cl_prim = Cl_prim[mask_lmin, ...]
    Cl_lens = Cl_lens[mask_lmin, ...]
    if Sc_approx is not None:
	Sc_approx = Sc_approx[mask_lmin, ...]


    # Start evaluating Fisher                                                                                                                                                                               
    try:
        cov = np.linalg.inv(alg._mtmm(A_ev(x0[:-1]), invNlm, A_ev(x0[:-1])))
    except np.linalg.LinAlgError:
        cov = np.zeros_like(alg._mtmm(A_ev(x0[:-1]), invNlm, A_ev(x0[:-1])))
	good_idx = np.where(np.all(np.diagonal(alg._mtmm(A_ev(x0[:-1]), invNlm, A_ev(x0[:-1])), axis1=-1, axis2=-2), axis=-1))
        cov[good_idx] = np.linalg.inv(alg._mtmm(A_ev(x0[:-1]), invNlm[good_idx], A_ev(x0[:-1])))
    s = alg._mv(cov, alg._mtmv(A_ev(x0[:-1]), invNlm, data)) # True s                                                                                                                                       


    A_dB = alg._format_A_dB(A_dB_ev(x0[:-1]), x0[:-1], nblind) # mixing matrix derivatives                                                                                                                  

    Fisher = 0.5*alg.nonparam_biascorr_fisher_bruteforce(A_ev(x0[:-1]), s, x0[-1]*Cl_prim+Cl_lens, Sc_approx, A_dB, Cl_prim, comp_of_param, invNlm)
    w, v = np.linalg.eig(Fisher)
    print("eigenvalues : ", w)                                                                                                                                                                             
    cond = np.linalg.cond(Fisher)
    print("condition number : ", cond)

    return Fisher

def nonparam_biascorr_fisher_bruteforce(A, s, Sc, Sc_approx, A_dB, Sc_dr, invN=None):
    
    Fisher = np.zeros((len(A_dB)+1, len(A_dB)+1))

    try:
        cov = np.linalg.inv(_mtmm(A, invN, A))
    except np.linalg.LinAlgError:
        cov = np.zeros_like(_mtmm(A, invN, A))
        good_idx = np.where(np.all(np.diagonal(_mtmm(A, invN, A), axis1=-1, axis2=-2), axis=-1))
        cov[good_idx] = np.linalg.inv(_mtmm(A, invN[good_idx], A))

    Nc = cov[..., 0, 0]
    P = invN - _mmm(_mm(invN, A), cov, _mtm(A, invN))

    #Define CMB average of s*T(s)                                                                                                                                                                           
    sst_avg = np.zeros((Sc.shape[0], s.shape[1], s.shape[-1], s.shape[-1]))
    sst_avg[..., 0, 0] = Sc
    sst_avg[..., 1:, 1:] = _uvt(s[..., 1:], s[..., 1:])

    for i in np.arange(len(A_dB)):
        A_dB_i = A_dB[i]
        print('A_dB = ', A_dB_i)
        for j in np.arange(len(A_dB)):
            A_dB_j = A_dB[j]

            Nc_dB = -2.0*_mmm(cov, _mtmm(A_dB_i, invN, A), cov)[..., 0, 0]
            Nc_dB_prime = -2.0*_mmm(cov, _mtmm(A_dB_j, invN, A), cov)[..., 0, 0]

            with np.errstate(divide='ignore'):
                m1_dBdB = 2.0*np.trace(_mtmm(A_dB_i, P, _mm(A_dB_j, sst_avg)), axis1=-1, axis2=-2) - Nc_dB*Nc_dB_prime/((Sc+Nc)**2) + Nc_dB*Nc_dB_prime/((Sc_approx+Nc)**2)
                m2_dBdB = 2.0*(Sc-Sc_approx)/((Sc_approx+Nc)*(Sc+Nc))*_mmm(cov, _mmm(_mtmm(A, invN, A_dB_i), cov, _mtmm(A, invN, A_dB_j)) + _mmm(_mtmm(A, invN, A_dB_j), cov, _mtmm(A, invN, A_dB_i)) + _mm\
m(_mtmm(A, invN, A_dB_i), cov, _mtmm(A_dB_j, invN, A)) - _mtmm(A_dB_i, P, A_dB_j), cov)[..., 0, 0]
                m3_dBdB = 2.0/(Sc+Nc)*_mmm(cov, _mmm(_mtmm(A, invN, A_dB_i), sst_avg, _mtmm(A_dB_j, invN, A)), cov)[..., 0, 0]
            m1_dBdB[~np.isfinite(m1_dBdB)] = 0.
            m2_dBdB[~np.isfinite(m2_dBdB)] = 0.
            m3_dBdB[~np.isfinite(m3_dBdB)] = 0.
            Fisher[i, j] = np.sum(m1_dBdB+m2_dBdB+m3_dBdB)

    m_drdr = np.zeros(Sc.shape)
    with np.errstate(divide='ignore'):
        m_drdr = (Sc_dr/(Sc+Nc))**2
    m_drdr[~np.isfinite(m_drdr)] = 0.
    Fisher[-1, -1] = np.sum(m_drdr)
    print(Fisher)

    w, v = np.linalg.eig(Fisher)
    print("eigenvalues : ", w)
    cond = np.linalg.cond(Fisher)
    print("condition number : ", cond)

    return Fisher

