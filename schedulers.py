from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import math
import torch


def get_cosine_with_hard_restarts_schedule_with_warmup_and_decay(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    num_cycles=1, 
    decay_factor=0.9
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cycle_progress = progress * num_cycles % 1  # Progress within the current cycle
        decay = decay_factor ** (progress * num_cycles)  # Apply decay per cycle

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycle_progress))) * decay
    
    return LambdaLR(optimizer, lr_lambda)