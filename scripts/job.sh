#!/bin/bash
#SBATCH -A m1727_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --array=0-9
#SBATCH -t 06:00:00
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1


module load tensorflow/2.6.0

cd /pscratch/sd/d/dlan/result_paper_IA_0/jac_ps_multiscale/

python /global/homes/d/dlan/DifferentiableHOS/scripts/compute_statistics.py  --filename=res_maps_0_$SLURM_ARRAY_TASK_ID --Power_Spectrum=True --Aia=0.

#--Convergence_map=True
#--Power_Spectrum=True  
#--Peak_counts=True
#--l1_norm=True
