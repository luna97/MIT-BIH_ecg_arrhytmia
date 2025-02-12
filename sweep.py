
from trainer_pretrain import train
import lightning as L
import wandb

def main():
    L.seed_everything(42)
    run = wandb.init()
    train(wandb.config, run, wandb=True)

main()