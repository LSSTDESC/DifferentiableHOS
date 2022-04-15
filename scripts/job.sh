#!/bin/bash
#SBATCH -A m1727_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --array=0-9
#SBATCH -t 04:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1


module load tensorflow/2.6.0

cd /pscratch/sd/d/dlan/results_paper/jac_peakcounts/

python /global/homes/d/dlan/DifferentiableHOS/scripts/compute_jacobian.py  --filename=res_maps_$SLURM_ARRAY_TASK_ID --Peak_counts=True

#--Convergence_map=True
#--Power_Spectrum=True  
#--Peak_counts=True
#--l1_norm=True