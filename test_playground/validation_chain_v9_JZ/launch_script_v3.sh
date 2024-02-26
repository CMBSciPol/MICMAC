# conda activate non_param_silver
source /Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/.bash_profile

export path_directory_main=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v9_JZ/
export path_Python_script=$path_directory_main/config_MICMAC_v1b_longrun.py

# export file_ver=test_corr_fullsky_v1e
# export add_param_toml=test_add_corr_fullsky_v0e
# export file_ver=test_corr_fullsky_v1f
# export add_param_toml=test_add_corr_fullsky_v0f
# export file_ver=test_corr_fullsky_LiteBIRD_v1a
# export add_param_toml=test_add_corr_fullsky_LiteBIRD_v0a
# export file_ver=test_corr_fullsky_LB_v1b
# export add_param_toml=test_add_corr_fullsky_LB_v0b
export file_ver=test_corr_fullsky_LB_v1bc
export add_param_toml=test_add_corr_fullsky_LB_v0bb
export file_ver=test_corr_fullsky_LB_v1c
export add_param_toml=test_add_corr_fullsky_LB_v0c


export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
