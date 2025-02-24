# (submit.sh)
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

# activate conda env
source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
srun python3 pretrain.py --epochs 100 \
    --dropout 0.3 \
    --activation_fn relu \
    --batch_size 128 \
    --patch_size 128 \
    --random_shift \
    --embedding_size 1024 \
    --use_scheduler \
    --xlstm_depth=8 \
    --lr 0.0001 \
    --wd 0.0001 \
    --pretrain_with_code15 \
    --multi_token_prediction \
    --nk_clean \
    --data_folder_code15 /media/Volume/data/CODE15/unlabeled_records_360_nkclean \
    --loss_type grad \
    --wandb_log