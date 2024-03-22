# conda activate micmac_env

# export path_local_MICMAC=/Users/arizzieri/APC/PhD/cmb_work/comp_sep/Non_param/MICMAC/
export path_local_MICMAC=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/

export path_directory_main=$path_local_MICMAC/test_playground/Validation_Classical_Gibbs/
export path_Python_script=$path_directory_main/script_v109_v4b.py

# export file_ver=class_fullsky_1freq_nofg_v1a2
# export add_param_toml=add_class_fullsky_f1freq_nofg_v1a

# export file_ver=class_cutsky_1freq_nofg_v1a5
# export add_param_toml=add_class_cutsky_f1freq_nofg_v1a

export file_ver=class_cutsky_1freq_nofg_v1cc
export add_param_toml=add_class_cutsky_f1freq_nofg_v1c

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
