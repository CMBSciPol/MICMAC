#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=micmac_v100
#SBATCH --partition quiet

#SBATCH --mail-user=rizzieri@apc.in2p3.fr
#SBATCH --mail-type=ALL

#SBATCH --output=micmac_v100.out


conda activate /work/rizzieri/.conda/envs/micmac_env

export path_directory_main=/work/rizzieri/MICMAC/test_playground/validation_chain_spatialvariability_v3

export path_Python_script=$path_directory_main/script_v109_v3.py


export file_ver=corr_fullsky_SO_d1s1_nside1_v109_v1
export add_param_toml=add_corr_fullsky_SO_d1s1_nside1_v1

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log
