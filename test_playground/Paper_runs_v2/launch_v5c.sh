# conda activate micmac_env

# export path_local_MICMAC=/Users/arizzieri/APC/PhD/cmb_work/comp_sep/Non_param/MICMAC/
export path_local_MICMAC=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/

export path_directory_main=$path_local_MICMAC/test_playground/Paper_runs_v2/
export path_Python_script=$path_directory_main/script_v109_v5.py

export file_ver=corr_cutsky_SO_d7s0_nside0_v1091b_v1a
export add_param_toml=add_corr_cutsky_SO_d7s0_nside0_v1a

export file_ver=corr_cutsky_LB_d0s0_r0_v1k
export add_param_toml=add_corr_cutsky_LB_d0s0_v1k

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
