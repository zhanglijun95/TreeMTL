#!/bin/bash
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --mem=10G              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
####efefSBATCH --cpus-per-task=4
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=FAIL
##### #SBATCH --mail-user=lijunzhang@cs.umass.edu

hostname
python combined_est_verify.py "$@"

sleep 1
exit
