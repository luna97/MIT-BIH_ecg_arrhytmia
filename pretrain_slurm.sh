#!/bin/bash -l

#SBATCH --job-name=pretrain_xlstm
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --time=1-00:00:00
#SBATCH -o ./slurm_output_%j_%x.out # STDOUT

source /home/$USER/.bashrc
conda activate xlstm_pretrained


# run script from above
srun python3 -u pretrain.py --epochs 150 \
    --dropout 0.2 \
    --activation_fn relu \
    --batch_size 512 \
    --patch_size 64 \
    --random_shift \
    --embedding_size 1024 \
    --use_scheduler \
    --lr 0.0001 \
    --wd 0.0001 \
    --pretrain_datasets code15 mimic \
    --nk_clean \
    --leads I II III aVR aVL aVF V1 V2 V3 V4 V5 V6 \
    --normalize
    --random_drop_leads 0.2
    --xlstm_config m s m m m m m m m s m m m m m m m s m m m m m m
    --loss_type grad \
    --wandb_log
