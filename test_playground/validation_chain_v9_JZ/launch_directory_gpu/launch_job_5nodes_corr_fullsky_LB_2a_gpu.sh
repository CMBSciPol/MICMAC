#!/bin/bash
#SBATCH --job-name=corr_fullsky_LB_2a_5nodes_gpu
#SBATCH --account=nih@v100          # use CPU allocation
#SBATCH --ntasks=5                 # nbr of MPI processes
#SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=40          # nbr of OpenMP threads
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --mail-user=magdy.morshed.fr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00            # default: 10 minutes on cpu_p1


# go to submit directory (where the .slurm file is)
cd ${SLURM_SUBMIT_DIR}

# clean up modules that have been launched in interactive session
module purge

# load intel modules
# module load intel-all/2021.9.0

# load python modules
module load python

#conda activate /gpfswork/rech/nih/commun/micmac_soft/micmac_env
source /gpfswork/rech/nih/ube74zo/MICMAC/.bash_env_gpu

export PYSM_LOCAL_DATA=/gpfswork/rech/nih/commun/micmac_soft/pysm-data

echo $PYSM_LOCAL_DATA

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# OpenMP binding
export OMP_PLACES=cores



export VER=corr_fullsky_LB_v104_v2a_gpu
export additional_config_file=add_gpu_corr_fullsky_LB_v2a.toml

export SRC_PATH=/gpfswork/rech/nih/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v9_JZ

srun python $SRC_PATH/config_MICMAC_v1b_longrun.py $additional_config_file  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log

echo "Run finished !"
