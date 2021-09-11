#!/bin/bash
#SBATCH -N 1
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --mail-user=denise.lanzieri@cea.fr
#SBATCH -t 00:50:00


module load python3/3.8-anaconda-2020.11

python /global/homes/d/dlan/DifferentiableHOS/scripts/compute_jacobian.py
