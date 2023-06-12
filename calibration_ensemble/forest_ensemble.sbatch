#!/bin/bash
#SBATCH -p skylake
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --array=1-12

julia --project=../ --threads=14 run_model.jl output_v3 5 $SLURM_ARRAY_TASK_ID 
