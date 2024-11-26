import jax.numpy as jnp
import numpy as np
import toml

from micmac.external.fgbuster import get_instrument
from micmac.foregrounds.templates import get_nodes_b, tree_spv_config
from micmac.likelihood.harmonic import HarmonicMicmacSampler
from micmac.likelihood.pixel import MicmacSampler
from micmac.noise.noisecovar import get_noise_covar_extended, get_true_Cl_noise
from micmac.toolbox.utils import get_instr

__all__ = [
    'create_MicmacSampler_from_dictionnary',
    'create_MicmacSampler_from_toml_file',
    'create_HarmonicMicmacSampler_from_dictionnary',
    'create_HarmonicMicmacSampler_from_toml_file',
    'create_HarmonicMicmacSampler_from_MicmacSampler_obj',
    'create_MicmacSampler_from_HarmonicMicmacSampler_obj',
]


def create_MicmacSampler_from_dictionnary(dictionary_parameters, path_file_spv=''):
    """
    Create a MicmacSampler object from:
    * the path of a toml file: params for the sims and for the sampling
    * the path of a spv file: params for addressing spatial variability

    Parameters
    ----------
    dictionary_parameters : dictionary
        dictionary for the main options of MicmacSampler
    path_file_spv : str
        path to the yaml file for the spatial variability options

    Returns
    -------
    MICMAC_Sampler_obj: MicmacSampler
        the MicmacSampler object created from the toml file with the spatial variability from the yaml file
    """

    ## Getting the instrument and the noise covariance
    if dictionary_parameters['instrument_name'] != 'customized_instrument':  ## TODO: Improve a bit this part
        instrument = get_instrument(dictionary_parameters['instrument_name'])

    else:
        instrument = get_instr(dictionary_parameters['frequency_array'], dictionary_parameters['depth_p'])
        del dictionary_parameters['depth_p']

    dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
    dictionary_parameters['freq_inverse_noise'] = get_noise_covar_extended(
        instrument['depth_p'], dictionary_parameters['nside']
    )

    ## Spatial variability (spv) params
    n_fgs_comp = dictionary_parameters['n_components'] - 1
    # total number of params in the mixing matrix for a specific pixel
    n_betas = (
        np.shape(dictionary_parameters['frequency_array'])[0] - len(dictionary_parameters['pos_special_freqs'])
    ) * (n_fgs_comp)
    # Read or create spv config
    root_tree = tree_spv_config(path_file_spv, n_betas, n_fgs_comp, print_tree=True)
    dictionary_parameters['spv_nodes_b'] = get_nodes_b(root_tree)

    ## Getting the covariance of Bf from toml file
    if 'step_size_Bf_1' in dictionary_parameters and 'step_size_Bf_2' in dictionary_parameters:
        n_frequencies = len(dictionary_parameters['frequency_array'])
        col_dim_Bf = n_frequencies - len(dictionary_parameters['pos_special_freqs'])

        dictionary_parameters['covariance_Bf'] = np.zeros(
            (col_dim_Bf * n_fgs_comp, col_dim_Bf * n_fgs_comp)
        )  # Creating the covariance matrix for Bf

        np.fill_diagonal(
            dictionary_parameters['covariance_Bf'][:col_dim_Bf, :col_dim_Bf],
            dictionary_parameters['step_size_Bf_1'] ** 2,
        )  # Filling diagonal with step_size_Bf_1 for first foreground component
        np.fill_diagonal(
            dictionary_parameters['covariance_Bf'][col_dim_Bf : 2 * col_dim_Bf, col_dim_Bf : 2 * col_dim_Bf],
            dictionary_parameters['step_size_Bf_2'] ** 2,
        )  # Filling diagonal with step_size_Bf_2 for second foreground component

        del dictionary_parameters['step_size_Bf_1']
        del dictionary_parameters['step_size_Bf_2']

    return MicmacSampler(**dictionary_parameters)


def create_MicmacSampler_from_toml_file(path_toml_file, path_file_spv=''):
    """
    Create a MicmacSampler object from:
    * the path of a toml file: params for the sims and for the sampling
    * the path of a spv file: params for addressing spatial variability

    Parameters
    ----------
    path_toml_file : str
        path to the toml file for the main options of MicmacSampler
    path_file_spv : str
        path to the yaml file for the spatial variability options

    Returns
    -------
    MICMAC_Sampler_obj: MicmacSampler
        the MicmacSampler object created from the toml file with the spatial variability from the yaml file
    """
    ### Opening first the toml file for the simulations and sampling, to create the MicmacSampler object
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    return create_MicmacSampler_from_dictionnary(dictionary_parameters, path_file_spv=path_file_spv)


def create_HarmonicMicmacSampler_from_dictionnary(dictionary_parameters, path_file_spv):
    """
    Create a HarmonicMicmacSampler object from the path of a toml file and the yaml file for spatial variability

    Parameters
    ----------
    dictionary_parameters : dictionary
        dictionary for the main options of HarmonicMicmacSampler
    path_file_spv : str
        path to the yaml file for the spatial variability options

    Returns
    -------
    HarmonicMicmacSampler_obj : HarmonicMicmacSampler
        HarmonicMicmacSampler object
    """

    if dictionary_parameters['instrument_name'] != 'customized_instrument':
        instrument = get_instrument(dictionary_parameters['instrument_name'])
    else:
        instrument = get_instr(dictionary_parameters['frequency_array'], dictionary_parameters['depth_p'])
        del dictionary_parameters['depth_p']

    dictionary_parameters['frequency_array'] = jnp.array(instrument['frequency'])
    dictionary_parameters['freq_noise_c_ell'] = get_true_Cl_noise(
        jnp.array(instrument['depth_p']), dictionary_parameters['lmax']
    )[..., dictionary_parameters['lmin'] :]

    ## Spatial variability (spv) params
    n_fgs_comp = dictionary_parameters['n_components'] - 1
    # total number of params in the mixing matrix for a specific pixel
    n_betas = (
        np.shape(dictionary_parameters['frequency_array'])[0] - len(dictionary_parameters['pos_special_freqs'])
    ) * (n_fgs_comp)
    # Read or create spv config
    root_tree = tree_spv_config(path_file_spv, n_betas, n_fgs_comp, print_tree=True)
    dictionary_parameters['spv_nodes_b'] = get_nodes_b(root_tree)

    ## Getting the covariance of Bf from toml file
    if 'step_size_Bf_1' in dictionary_parameters and 'step_size_Bf_2' in dictionary_parameters:
        n_frequencies = len(dictionary_parameters['frequency_array'])
        col_dim_Bf = n_frequencies - len(dictionary_parameters['pos_special_freqs'])

        dictionary_parameters['covariance_Bf'] = np.zeros(
            (col_dim_Bf * n_fgs_comp, col_dim_Bf * n_fgs_comp)
        )  # Creating the covariance matrix for Bf

        np.fill_diagonal(
            dictionary_parameters['covariance_Bf'][:col_dim_Bf, :col_dim_Bf],
            dictionary_parameters['step_size_Bf_1'] ** 2,
        )  # Filling diagonal with step_size_Bf_1 for first foreground component
        np.fill_diagonal(
            dictionary_parameters['covariance_Bf'][col_dim_Bf : 2 * col_dim_Bf, col_dim_Bf : 2 * col_dim_Bf],
            dictionary_parameters['step_size_Bf_2'] ** 2,
        )  # Filling diagonal with step_size_Bf_2 for second foreground component

        del dictionary_parameters['step_size_Bf_1']
        del dictionary_parameters['step_size_Bf_2']

    return HarmonicMicmacSampler(**dictionary_parameters)


def create_HarmonicMicmacSampler_from_toml_file(path_toml_file, path_file_spv=''):
    """
    Create a MicmacSampler object from:
    * the path of a toml file: params for the sims and for the sampling
    * the path of a spv file: params for addressing spatial variability

    Parameters
    ----------
    path_toml_file : str
        path to the toml file for the main options of MicmacSampler
    path_file_spv : str
        path to the yaml file for the spatial variability options

    Returns
    -------
    MICMAC_Sampler_obj: MicmacSampler
        the MicmacSampler object created from the toml file with the spatial variability from the yaml file
    """
    ### Opening first the toml file for the simulations and sampling, to create the MicmacSampler object
    with open(path_toml_file) as f:
        dictionary_parameters = toml.load(f)
    f.close()

    return create_HarmonicMicmacSampler_from_dictionnary(dictionary_parameters, path_file_spv=path_file_spv)


def create_HarmonicMicmacSampler_from_MicmacSampler_obj(MICMAC_sampler_obj, depth_p_array=None):
    """
    Create a HarmonicMicmacSampler object from a MicmacSampler object

    Parameters
    ----------
    MICMAC_sampler_obj : MicmacSampler
        MicmacSampler object
    depth_p_array : array[float] of dimensions [n_pix]
        depth_p_array for the noise power spectrum

    Returns
    -------
    HarmonicMicmacSampler_obj : HarmonicMicmacSampler
        HarmonicMicmacSampler object
    """

    arguments_HarmonicMicmacSampler = inspect.getfullargspec(HarmonicMicmacSampler).args

    dictionary_parameters = dict()
    for attr in arguments_HarmonicMicmacSampler:
        if attr in MICMAC_sampler_obj:
            dictionary_parameters[attr] = getattr(MICMAC_sampler_obj, attr)

    if MICMAC_sampler_obj['freq_noise_c_ell'] is None:
        assert depth_p_array is not None
        dictionary_parameters['freq_noise_c_ell'] = get_true_Cl_noise(depth_p_array, MICMAC_sampler_obj.lmax)[
            ..., MICMAC_sampler_obj.lmin :
        ]

    return HarmonicMicmacSampler(**dictionary_parameters)


def create_MicmacSampler_from_HarmonicMicmacSampler_obj(HarmonicMicmac_sampler_obj, depth_p_array=None):
    """
    Create a MicmacSampler object from a HarmonicMicmacSampler object

    Parameters
    ----------
    HarmonicMicmac_sampler_obj : HarmonicMicmacSampler
        HarmonicMicmacSampler object
    depth_p_array : array[float] of dimensions [n_pix]
        depth_p_array for the noise power spectrum

    Returns
    -------
    MICMAC_sampler_obj : MicmacSampler
        MicmacSampler object
    """

    arguments_MicmacSampler = inspect.getfullargspec(MicmacSampler).args

    dictionary_parameters = dict()
    for attr in arguments_MicmacSampler:
        if attr in HarmonicMicmac_sampler_obj:
            dictionary_parameters[attr] = getattr(HarmonicMicmac_sampler_obj, attr)

    if depth_p_array is not None:
        dictionary_parameters['freq_inverse_noise'] = get_noise_covar_extended(
            depth_p_array, HarmonicMicmac_sampler_obj.nside
        )  # MICMAC_obj.freq_inverse_noise
    else:
        dictionary_parameters['freq_inverse_noise'] = None
        print(
            'No depth_p_array provided, freq_inverse_noise set to None, the Gibbs sampling cannot be launched if freq_inverse_noise is set to None',
            flush=True,
        )

    return MicmacSampler(**dictionary_parameters)
