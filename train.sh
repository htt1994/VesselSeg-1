#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=3:00
#SBATCH --job-name=DenseNetPBR_t1
#SBATCH --output=$SCRATCH/output/DenseNetPBR_t1.txt
# Emails me when job starts, ends or fails
#SBATCH --mail-user=address@email.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-wanglab

#load any required modules
module load python/3.6.0
# activate the virtual environment
source ~/<ENV>/bin/activate

# run a training session
srun python <LOCATION>
