from typing import Tuple

import torch
import torch.nn as nn

__all__ = ['Optimizer', 'Scheduler']


class Optimizer:
    def __init__(self):
        pass
    
    def get(self, model: nn.Module, optimizer: str, lr: float, wd: float = 0., momentum: float = 0.5,
            betas: Tuple[float, float] = (0.9, 0.999)):

        if optimizer.lower() == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        elif optimizer.lower() == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        elif optimizer.lower() == 'none':
            optim = Optimizer()
        else:
            raise ValueError("Optimizer {} not supported".format(optimizer))
        return optim


class Scheduler:
    def __init__(self):
        pass
    
    def get( self, lr_scheduler: str, optimizer: torch.optim.Optimizer, step_size: int, gamma: float = 0.5):
        if lr_scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        elif lr_scheduler.lower() == 'none':
            scheduler = Scheduler()
        else:
            raise ValueError("lr_scheduler {} not supported".format(lr_scheduler))
        return scheduler
