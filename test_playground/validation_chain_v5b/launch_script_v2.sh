# conda activate non_param_silver
source /Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/.bash_profile

export path_directory_main=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v5b/
export path_Python_script=$path_directory_main/config_MICMAC_v1b2_longrun.py

export file_ver=invWishart_fullsky_v104_v1c
export add_param_toml=add_invWishart_fullsky_v1c

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
