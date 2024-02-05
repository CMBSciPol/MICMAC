conda activate non_param_silver

export path_directory_main=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/test_playground/validation_chain_v5b
export path_Python_script=$path_directory_main/config_MICMAC_v1b_longrun.py

export additional_params=$path_directory_main/save_directory/invWishart_fullsky_v1a.toml

mpirun -n 1 python $path_Python_script $additional_params