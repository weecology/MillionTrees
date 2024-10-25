#!/bin/bash
#SBATCH --job-name=MillionTrees_DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=ben.weinstein@weecology.org # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI r
#SBATCH --cpus-per-task=2
#SBATCH --mem=200GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_MillionTrees_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_MillionTrees_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

export PATH="$PATH:/orange/ewhite/b.weinstein/miniconda3_new/bin"
source activate MillionTrees

export COMET_PROJECT_NAME=milliontrees
export COMET_WORKSPACE=bw4sz

#Add to path
export PYTHONPATH=$PYTHONPATH:/home/b.weinstein/MillionTrees

python DeepForest.py