#!/bin/bash
#SBATCH --job-name=biased_0a
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
#SBATCH --cpus-per-task=10          # nbr of OpenMP threads
#SBATCH --hint=nomultithread       # 1 thread / physical core (no hyperthreading)
#SBATCH --mail-user=magdy.morshed.fr@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00            # default: 10 minutes on cpu_p1

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
# source /linkhome/rech/genkqu01/ube74zo/MICMAC/.bash_profile

# export PYSM_LOCAL_DATA=/gpfswork/rech/nih/commun/micmac_soft/pysm-data

echo $PYSM_LOCAL_DATA

#echo launched commands
# set -x

# number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# OpenMP binding
export OMP_PLACES=cores



# srun python /linkhome/rech/genkqu01/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v6_JZ/very_cheap_biased_unmasked_fullchain_v100_Gibbs_withr_SO_64_v1a_longrun.py  1> logs/log_biased_unmasked_full_v101_Gchain_SO_64_v2a.log 2> errs/err_biased_unmasked_full_v101_Gchain_SO_64_v2a.log
# srun python /linkhome/rech/genkqu01/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v6_JZ/very_cheap_biased_unmasked_fullchain_v100_Gibbs_withr_SO_64_v1a_longrun.py  1> logs/log_biased_unmasked_full_v101_Gchain_SO_64_v2c.log 2> errs/err_biased_unmasked_full_v101_Gchain_SO_64_v2c.log

# export VER=corr_masked_full_v100_Gchain_SO_64_3ab
# export VER=corr_unmasked_full_v101_Gchain_SO_64_v0a
export VER=biased_masked_full_v102_Gchain_SO_64_v0a

export SRC_PATH=/linkhome/rech/genkqu01/ube74zo/MICMAC/MICMAC/test_playground/validation_chain_v7_JZ
# srun python $SRC_PATH/very_cheap_corr_unmasked_fullchain_v101_Gibbs_withr_SO_64_v0a_longrun.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log
srun python $SRC_PATH/very_cheap_biased_masked_fullchain_v102_Gibbs_withr_SO_64_v0a_longrun.py  1> $SRC_PATH/logs/log_$VER.log 2> $SRC_PATH/errs/err_$VER.log


echo "Run finished !"
