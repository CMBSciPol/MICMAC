#!/bin/bash
#SBATCH --job-name=class_v1e
#SBATCH --account=nih@cpu          # use CPU allocation
#SBATCH --qos=qos_cpu-dev          # dev qos (10 jobs, 2h max.)
#SBATCH --mail-user=magdy.morshed.fr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=4
#SBATCH --ntasks=4                 # nbr of MPI processes
#SBATCH --ntasks-per-node=1       # Nombre de processus MPI par noeud
#SBATCH --cpus-per-task=40          # nbr of OpenMP threads
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --time=02:00:00            # default: 10 minutes on cpu_p1

# go to submit directory (where the .slurm file is)
cd ${SLURM_SUBMIT_DIR}

# clean up modules that have been launched in interactive session
module purge

# load intel modules
# module load intel-all/2021.9.0

# load python modules
module load python

source /gpfswork/rech/nih/ube74zo/MICMAC/.bash_env


export PYSM_LOCAL_DATA=/gpfswork/rech/nih/commun/micmac_soft/pysm-data

echo $PYSM_LOCAL_DATA

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# OpenMP binding
export OMP_PLACES=cores



export VER=class_cutsky_1freq_nofg_v1f3
export additional_config_file=add_class_cutsky_f1freq_nofg_v1f.toml

export SRC_PATH=/gpfswork/rech/nih/ube74zo/MICMAC/MICMAC/test_playground/Validation_Classical_Gibbs

srun python $SRC_PATH/script_v109_v4b.py $additional_config_file  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log

echo "Run finished !"
