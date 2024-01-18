#!/bin/bash
#SBATCH --job-name=Iwish_0
#SBATCH --account=nih@cpu          # use CPU allocation
#SBATCH --qos=qos_cpu-t3          # dev qos (10 jobs, 2h max.)
#SBATCH --mail-user=magdy.morshed.fr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1                 # nbr of MPI processes
#SBATCH --cpus-per-task=40          # nbr of OpenMP threads
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --time=20:00:00            # default: 10 minutes on cpu_p1

# go to submit directory (where the .slurm file is)
cd ${SLURM_SUBMIT_DIR}

# clean up modules that have been launched in interactive session
module purge

# load intel modules
module load intel-all/2021.9.0

# load python modules
module load python
#conda activate /gpfswork/rech/nih/commun/micmac_soft/micmac_env
source /linkhome/rech/genkqu01/ube74zo/MICMAC/.bash_profile

echo $PYSM_LOCAL_DATA

# echo launched commands
# set -x

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# OpenMP binding
export OMP_PLACES=cores

# srun python /linkhome/rech/genkqu01/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v6_JZ/very_cheap_Iwish_biased_unmasked_fullchain_v100_Gibbs_withr_SO_64_v1a_longrun.py  1> logs/log_Iwish_biased_unmasked_full_v100_Gchain_SO_64_v3a.log 2> errs/err_Iwish_biased_unmasked_full_v100_Gchain_SO_64_v3a.log

export VER=Iwish_biased_unmasked_full_v100_Gchain_SO_64_v3ab

export SRC_PATH=/linkhome/rech/genkqu01/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v6_JZ
srun python $SRC_PATH/very_cheap_Iwish_biased_unmasked_fullchain_v100_Gibbs_withr_SO_64_v1c_longrun.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log

echo "Run finished !"