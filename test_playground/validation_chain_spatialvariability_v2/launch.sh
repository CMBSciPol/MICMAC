# conda activate micmac_env

export path_directory_main=/Users/arizzieri/APC/PhD/cmb_work/comp_sep/Non_param/MICMAC/test_playground/validation_chain_spatialvariability_v2/
export path_Python_script=$path_directory_main/script_v108_v1.py

export file_ver=corr_fullsky_SO_v108_v1b
export add_param_toml=add_corr_fullsky_SO_v1b

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
