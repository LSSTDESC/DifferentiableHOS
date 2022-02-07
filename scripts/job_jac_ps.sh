#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --array=0-20
#SBATCH -t 02:30:00
#SBATCH --mail-user=denise.lanzieri@cea.fr

conda deactivate
module load python/3.8-anaconda-2020.11

cd /global/cscratch1/sd/dlan/jacobian_ps/


python /global/homes/d/dlan/DifferentiableHOS/scripts/compute_jacobian_ps.py  --filename=res_ps_$SLURM_ARRAY_TASK_ID