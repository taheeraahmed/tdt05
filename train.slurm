#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --job-name="tdt05-ssl"
#SBATCH --output=idun/train.out
#SBATCH --gres=gpu:1

module load Anaconda3/2022.05
pip install -r requirements.txt
jupyter notebook --no-browser