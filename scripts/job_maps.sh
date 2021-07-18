#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --array=0-63
#SBATCH -t 01:00:00
#SBATCH --mail-user=denise.lanzieri@cea.fr

module load python3/3.8-anaconda-2020.11

cd /global/cscratch1/sd/dlan/plot_paper2

python /global/homes/d/dlan/DifferentiableHOS/scripts/compute_maps.py  --filename=res_maps_$SLURM_ARRAY_TASK_ID
