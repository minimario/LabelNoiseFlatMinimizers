source ~/.bashrc
cd /om2/user/gua/Documents/LabelNoiseFlatMinimizers
conda activate /scratch2/weka/tenenbaum/gua/anaconda3/envs/smoothing
# wandb disabled
wandb enabled
    
rho=0.0
wandb_run_name="labelnoise_sgd"
python3 train.py \
    --lr 5 \
    --smoothing 0.2 \
    --label_noise \
    --rho $rho \
    --init init/fullbatch.pt \
    --momentum 0 \
    --name $wandb_run_name \
    --save output \
    --batch_size 256 \
    --callbacks