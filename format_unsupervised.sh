#!/bin/bash
#SBATCH --job-name=MillionTrees_DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/Unsupervised_MillionTrees_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/Unsupervised_MillionTrees_%j.err

export PATH="$PATH:/orange/ewhite/b.weinstein/miniconda3/bin"
source activate MillionTrees

#Add to path
export PYTHONPATH=$PYTHONPATH:/home/b.weinstein/MillionTrees
python data_prep/package_datasets.py