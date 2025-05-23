#!/bin/bash
#SBATCH -J llmcompass 
#SBATCH -p batch_long.q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time 7-00:00:00
#SBATCH --error "SLURM_%j_%x".err
#SBATCH --mail-user $USER@imec.be
#SBATCH --mail-type=ALL


source /imec/users/patel23/.bashrc
conda activate llmcompass

cd /imec/scratch/dtpatha/patel23/LLM_analytical_tools/LLMCompass
python -m ae.simRuns.param_sweeps_new
