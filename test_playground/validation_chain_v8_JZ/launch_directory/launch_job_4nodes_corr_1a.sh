#!/bin/bash
#SBATCH --job-name=corr_1a_4nodes
#SBATCH --account=nih@cpu          # use CPU allocation
#SBATCH --qos=qos_cpu-t3          # dev qos (10 jobs, 2h max.)
#SBATCH --mail-user=magdy.morshed.fr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=4                 # nbr of MPI processes
#SBATCH --ntasks-per-node=1       # Nombre de processus MPI par noeud
#SBATCH --cpus-per-task=40          # nbr of OpenMP threads
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --time=20:00:00            # default: 10 minutes on cpu_p1

# go to submit directory (where the .slurm file is)
cd ${SLURM_SUBMIT_DIR}

# clean up modules that have been launched in interactive session
module purge

# load intel modules
# module load intel-all/2021.9.0
echo "Test 0 !"
# load python modules
module load python
echo "Test 1 !"
#conda activate /gpfswork/rech/nih/commun/micmac_soft/micmac_env
source /linkhome/rech/genkqu01/ube74zo/MICMAC/.bash_profile
echo "Test 2 !"

export PYSM_LOCAL_DATA=/gpfswork/rech/nih/commun/micmac_soft/pysm-data

echo $PYSM_LOCAL_DATA

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Test 3 !"
# OpenMP binding
export OMP_PLACES=cores



# export VER=corr_masked_full_v100_Gchain_SO_64_3ab
# export VER=corr_unmasked_full_v101_Gchain_SO_64_v0a
export VER=corr_masked_v104_v1a_

export SRC_PATH=/linkhome/rech/genkqu01/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v8_JZ
# srun python $SRC_PATH/very_cheap_corr_unmasked_fullchain_v101_Gibbs_withr_SO_64_v0a_longrun.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log
# srun -n 4 python $SRC_PATH/config_MICMAC_v1a_longrun.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log
echo "Test 4 !"
srun python $SRC_PATH/config_MICMAC_v1a_longrun.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log
echo "Test 5 !"

echo "Run finished !"
