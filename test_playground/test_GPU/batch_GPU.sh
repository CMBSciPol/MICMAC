#!/bin/bash
#SBATCH --job-name=test_GPU
#SBATCH --account=nih@v100          # use CPU allocation
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH --partition=gpu_p4          # decommenter pour la partition gpu_p4 (GPU A100 40 Go)
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
#SBATCH --ntasks=1                 # nbr of MPI processes
#SBATCH --gres=gpu:4                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=40          # nbr of OpenMP threads
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --mail-user=magdy.morshed.fr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:15:00            # default: 10 minutes on cpu_p1

##SBATCH --qos=qos_gpu-t3          # dev qos (10 jobs, 2h max.)


# go to submit directory (where the .slurm file is)
cd ${SLURM_SUBMIT_DIR}

# clean up modules that have been launched in interactive session
module purge

# load intel modules
# module load intel-all/2021.9.0

# load python modules
module load python
conda activate /gpfswork/rech/nih/commun/micmac_soft/micmac_env_gpu
# source /gpfswork/rech/nih/ube74zo/MICMAC/.bash_env

export PYSM_LOCAL_DATA=/gpfswork/rech/nih/commun/micmac_soft/pysm-data

echo $PYSM_LOCAL_DATA

#echo launched commands
# set -x

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# OpenMP binding
export OMP_PLACES=cores


export VER=test

export SRC_PATH=/gpfswork/rech/nih/ube74zo/MICMAC/MICMAC/test_playground/test_GPU
srun python $SRC_PATH/script_test.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log


echo "Run finished !"
