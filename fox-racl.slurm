#!/bin/bash

#SBATCH --job-name=racl
#SBATCH --account=ec37
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

# print information (optional)
echo "submission directory: ${SUBMITDIR}"
ulimit -a
module list

# by default, pass on any remaining command-line options
# python3 -u train.py ${@}

# run code 
CUDA_VISIBLE_DEVICES=0 python3 train_racl_bert.py --task res14 --load 0
