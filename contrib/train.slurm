#!/bin/bash
#SBATCH -t 00:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gputest
#SBATCH --account=project_2004993
#SBATCH --cpus-per-task=32

module load pytorch/1.12

python train.py \
	--dataset shoe_reviews \
	--num-samples 1000 \
	--classification-baseline
