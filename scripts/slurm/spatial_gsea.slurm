#!/bin/bash
#SBATCH -p gpusaez
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --mem=250G
#SBATCH --job-name=dmgc-spatial-gsea
#SBATCH --output=logs/dmgc-spatial-gsea-%j.out

# Submit from the repository root after activating the appropriate environment.
source ~/.bashrc
conda activate cell2location
jupyter nbconvert --to notebook --execute \
  spatial/notebooks/analysis/05_gene_set_enrichment.ipynb \
  --output 05_gene_set_enrichment.executed.ipynb
