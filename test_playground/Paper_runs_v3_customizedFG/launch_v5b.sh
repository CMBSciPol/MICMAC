# conda activate micmac_env

export path_local_MICMAC=/Users/arizzieri/APC/PhD/cmb_work/comp_sep/Non_param/MICMAC/
# export path_local_MICMAC=/Users/mag/Documents/PHD1Y/Space_Work/Pixel_non_P2D/MICMAC/

export path_directory_main=$path_local_MICMAC/test_playground/Paper_runs_v3_customizedFG/
export path_Python_script=$path_directory_main/script_v109_v5b_customizedFG.py

export file_ver=corr_cutsky_d1s1custom1_LB_r0_v1b
export add_param_toml=add_corr_cutsky_LB_d1s1_v1b

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
