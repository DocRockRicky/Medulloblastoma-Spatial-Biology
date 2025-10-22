#!/bin/bash

#SBATCH --job-name="QC_Fil_Jack_Res"                     # job name
#SBATCH --partition=main
#SBATCH --nodes=1                              # node count
#SBATCH --ntasks=1                             # total number of tasks across all nodes
#SBATCH --cpus-per-task=12                      # cpu-cores per task
#SBATCH --mem=500G                             # total memory per node
#SBATCH --time=2-00:00                         # wall time DD-HH:MM
#SBATCH --output="/mount/mdtaylor2/ricardo/10XVisiumHD/Analysis/MDTAP0102/scripts/%x_%j.out"
#SBATCH --error="/mount/mdtaylor2/ricardo/10XVisiumHD/Analysis/MDTAP0102/scripts/%x_%j.err"
#SBATCH --mail-user=ricardo.gonzalez@bcm.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load required modules
module load R/4.4.1
module load gnu/gcc/12.3.0
module load imagemagick
module load openmpi/4.1.6
module load gnu/mp/6.2.0
module load cmake/3.27.6


# Initialize conda command for bash shell in this script
source /mount/mdtaylor2/ricardo/miniconda3/etc/profile.d/conda.sh

# Activate the 'spatial' conda environment
conda activate spatial

# Run the R script
Rscript /mount/mdtaylor2/ricardo/10XVisiumHD/Analysis/MDTAP0102/scripts/preprocess_spatial_polygons.R


