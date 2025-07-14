#!/bin/bash
#
#SBATCH --job-name=appletree
#SBATCH --output=appletree_case0_5.txt
#
#SBATCH --partition=dgx2q
#SBATCH --gres=gpu:a100:1
#
#SBATCH --ntasks=1

srun python /data/maia/gpxu/proj1/samatch/train_unimatch_medsam_F2_ft_both_acdc.py