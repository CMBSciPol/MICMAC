#!/bin/bash -l

##########################
# run on mbs on the fly  #
##########################
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=micmac_v100
#SBATCH --partition quiet

#SBATCH --mail-user=rizzieri@apc.in2p3.fr
#SBATCH --mail-type=ALL

#SBATCH --output=micmac_v100.out

conda activate /work/rizzieri/.conda/envs/micmac_env

export path_directory_main=/work/rizzieri/MICMAC/test_playground/validation_chain_Dante_Arianna
export path_Python_script=$path_directory_main/config_MICMAC_v1b_longrun.py

export file_ver=fullsky_Dante_Arianna_v100
export add_param_toml=test_add_corr_fullsky_v100

export additional_params=$path_directory_main/additional_params/$add_param_toml.toml

mpirun -n 1 python $path_Python_script $additional_params > $path_directory_main/logs/log_$file_ver.log

exit 0
