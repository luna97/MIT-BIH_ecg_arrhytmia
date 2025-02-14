
from pretrain import pretrain
from train import train
import lightning as L
import wandb

def main():
    L.seed_everything(42)
    run = wandb.init()
    if wandb.config.pretrain:
        pretrain(run.config, run, wandb=True)
    else:
        train(run.config, run, wandb=True)

main()