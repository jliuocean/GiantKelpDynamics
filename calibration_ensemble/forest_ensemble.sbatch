#!/bin/bash
#SBATCH -p skylake
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --array=1-12

julia --project=/nfs/st01/hpc-atmos-jrt51/js2430/KelpPhysics --threads=14 run_model.jl output 3 $SLURM_ARRAY_TASK_ID 
