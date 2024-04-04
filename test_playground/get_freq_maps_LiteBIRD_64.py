import numpy as np
from fgbuster.observation_helpers import *
from micmac import *


# General parameters
NSIDE = 64
cmb_model = 'c1'
fgs_model = 'd0s0'
model = cmb_model+fgs_model
noise = True
# noise = False
noise_seed = 42
instr_name = 'LiteBIRD'

# get instrument from public database
instrument = get_instrument(instr_name)

# get input freq maps
np.random.seed(noise_seed)
freq_maps_fgs = get_observation(instrument, fgs_model, nside=NSIDE, noise=noise)[:, 1:, :]   # keep only Q and U
# print("Shape for input frequency maps :", freq_maps.shape)

# get input cmb
input_cmb_maps = get_observation(instrument, cmb_model, nside=NSIDE, noise=False)[:, 1:, :]   # keep only Q and U
print("Shape for input cmb maps :", input_cmb_maps.shape)

freq_maps = freq_maps_fgs + input_cmb_maps