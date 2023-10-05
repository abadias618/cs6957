#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:2080ti
#SBATCH --time=2:00:00
#SBATCH --mem=10GB
#SBATCH --mail-user=u1419615@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_2-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nlp

OUT_DIR=/scratch/general/vast/u1419615/cs6957/assignment2/models
mkdir -p ${OUT_DIR}
python main.py
