#!/bin/bash

#SBATCH --mem=128G
#SBATCH -n64
#SBATCH --gres=gpu:2
#SBATCH --partition=amdgpu # partition name

module restore cone_seg_modules
wandb online
cd /mnt/personal/korcadav/eforce_cone_seg

python train.py --name RCI_256_Small_Mid_Transforms --rci
