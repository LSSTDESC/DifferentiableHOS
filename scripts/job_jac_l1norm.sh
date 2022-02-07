#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --array=0-50
#SBATCH -t 03:40:00
#SBATCH --mail-user=denise.lanzieri@cea.fr

conda deactivate
module load python/3.8-anaconda-2020.11

cd /global/cscratch1/sd/dlan/jacobian_l1norm/

python /global/homes/d/dlan/DifferentiableHOS/scripts/compute_jacobian_l1norm.py  --filename=res_l1_$SLURM_ARRAY_TASK_ID