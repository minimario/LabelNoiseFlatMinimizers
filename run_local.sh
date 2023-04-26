source ~/.bashrc
cd /om2/user/gua/Documents/LabelNoiseFlatMinimizers
conda activate /scratch2/weka/tenenbaum/gua/anaconda3/envs/smoothing
# wandb enabled
wandb disabled

# rho_values=(0.01 0.05 0.1 0.5 1 2)
rho=0.01
wandb_run_name="labelnoise_rs_rho_$rho"
python3 train.py \
    --lr 2 \
    --rho $rho \
    --init init/adv.pt \
    --momentum 0.9 \
    --name $wandb_run_name
