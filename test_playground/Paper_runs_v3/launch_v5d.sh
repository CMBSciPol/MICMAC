# conda activate micmac_env

# export path_local_MICMAC=/Users/arizzieri/APC/PhD/cmb_work/comp_sep/Non_param/MICMAC/
export path_local_MICMAC=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/

export path_directory_main=$path_local_MICMAC/test_playground/Paper_runs_v3/
export path_Python_script=$path_directory_main/script_v109_v5b.py

export file_ver=corr_cutsky_SO_d7s0_nside2_v1091c_r0_v1a5
export add_param_toml=add_corr_cutsky_SO_d7s0_nside2_v1a

export file_ver=corr_cutsky_SO_d7s0_nside2_v1091c_r0_v1b6
export add_param_toml=add_corr_cutsky_SO_d7s0_nside2_v1b

# export file_ver=corr_cutsky_SO_d7s0_nside4_v1091c_r0_v1a
# export add_param_toml=add_corr_cutsky_SO_d7s0_nside4_v1a

# export file_ver=corr_cutsky_SO_d1s1_nside0_v1091c_r0_v1ab
# export add_param_toml=add_corr_cutsky_SO_d1s1_nside0_v1a

# export file_ver=corr_cutsky_SO_d7s0_nside2_v1091c_r0_v1c2
# export add_param_toml=add_corr_cutsky_SO_d7s0_nside2_v1c




export file_ver=corr_cutsky_SO_d0s0_nside0_v1091c_r0_v1b
export add_param_toml=add_corr_cutsky_SO_d0s0_nside0_v1c

export file_ver=corr_cutsky_LB_d0s0_nside0_v1091c_r0_v1b
export add_param_toml=add_corr_cutsky_LB_d0s0_nside0_v1b


export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
