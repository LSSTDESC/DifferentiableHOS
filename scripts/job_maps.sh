#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --array=0-63
#SBATCH -t 01:00:00

module load tensorflow/intel-2.4.1

cd /global/u2/f/flanusse/scratch/Diff

python ~/repo/DifferentiableHOS/scripts/compute_maps.py  --filename=res_maps_$SLURM_ARRAY_TASK_ID