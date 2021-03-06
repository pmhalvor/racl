#!/bin/bash

#SBATCH --job-name=pmhalvor/fgsa
#SBATCH --account=nn9851k
#SBATCH --mail-type=FAIL
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks-per-node=2
#SBATCH --partition=accel
#SBATCH --gres=gpu:1

# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module use -a /cluster/projects/nn9851k/software/easybuild/install/modules/all/
module load TensorFlow/1.13.1-fosscuda-2019a-Python-3.7.2

# manually install correct scikit-learn 
# pip install scikit-learn==0.22.2 --user

# old moduyles for IN5550
# module load NLPL-nlptools/2021.01-gomkl-2019b-Python-3.7.4
# module load NLPL-PyTorch/1.6.0-gomkl-2019b-Python-3.7.4
# module load NLPL-transformers/4.2.2-gomkl-2019b-Python-3.7.4

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
ulimit -a
module list

# by default, pass on any remaining command-line options
# python3 -u train.py ${@}

# run code 
CUDA_VISIBLE_DEVICES=0 python3 train_racl_bert.py --task res14 --load 0