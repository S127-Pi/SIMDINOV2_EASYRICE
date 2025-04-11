#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --error=/lustrefs/disk/project/pc701043-embedd/SIMDINOV2_EASYRICE/MPIND1/%j_0_log.err
#SBATCH --gpus-per-node=3
#SBATCH --job-name=dinov2:train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --open-mode=append
#SBATCH --output=/lustrefs/disk/project/pc701043-embedd/SIMDINOV2_EASYRICE/MPIND1/%j_0_log.out
#SBATCH --partition=gpu
#SBATCH --signal=USR2@120
#SBATCH --time=300
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /lustrefs/disk/project/pc701043-embedd/SIMDINOV2_EASYRICE/MPIND1/%j_%t_log.out --error /lustrefs/disk/project/pc701043-embedd/SIMDINOV2_EASYRICE/MPIND1/%j_%t_log.err /home/smolling/.conda/envs/torch/bin/python3 -u -m submitit.core._submit /lustrefs/disk/project/pc701043-embedd/SIMDINOV2_EASYRICE/MPIND1
