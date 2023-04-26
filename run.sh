#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=8GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=2GB
#SBATCH --array=0-5
#SBATCH -p tenenbaum
#SBATCH -t 168:00:00

source ~/.bashrc
cd /om2/user/gua/Documents/LabelNoiseFlatMinimizers
conda activate /scratch2/weka/tenenbaum/gua/anaconda3/envs/smoothing
wandb enabled

rho_values=(0.01 0.05 0.1 0.5 1 2)
rho=${rho_values[$SLURM_ARRAY_TASK_ID]}
wandb_run_name="labelnoise_rs_rho_$rho"
python3 train.py \
    --lr 2 \
    --rho $rho \
    --init init/adv.pt \
    --name $wandb_run_name
