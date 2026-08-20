#!/bin/bash
#SBATCH -p gpusaez
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --mem=120G
#SBATCH --job-name=dmgc-spatial
#SBATCH --output=logs/dmgc-spatial-%j.out

# Submit from the repository root after activating the appropriate environment.
source ~/.bashrc
conda activate multi2
python -m scripts.train --task spatial --mode train "$@"
