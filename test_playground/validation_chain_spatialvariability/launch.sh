# conda activate micmac_env

export path_directory_main=/Users/arizzieri/APC/PhD/cmb_work/comp_sep/Non_param/MICMAC/test_playground/validation_chain_spatialvariability/
export path_Python_script=$path_directory_main/script.py

export file_ver=fullsky_spv0_v1
export add_param_toml=add_spv_v1

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
