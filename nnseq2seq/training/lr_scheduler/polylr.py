from typing import List, Dict, Tuple
import math
from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


class WarmupCosineLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, current_step: int = None,
            min_lr: float=1e-7,
            T_multi: int=1,
            N_cycle: int=1,
            warmup_epochs: int=10,
            warmup_initial_lr: float=1e-7,
        ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.T_multi = T_multi
        self.N_cycle = N_cycle
        self.warmup_epochs = warmup_epochs
        self.warmup_initial_lr = warmup_initial_lr
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step<=self.warmup_epochs:
            new_lr = self.warmup_initial_lr + (self.initial_lr - self.warmup_initial_lr) * (current_step / self.warmup_epochs)
        else:
            if self.T_multi==1:
                T_i = (self.max_steps-self.warmup_epochs)/self.N_cycle
                T_cur = (current_step-self.warmup_epochs)%T_i
            elif self.T_multi>1:
                T0 = (self.max_steps-self.warmup_epochs)*(self.T_multi-1)/(self.T_multi**self.N_cycle-1)
                n_i = math.floor(math.log(1+(current_step-self.warmup_epochs)*(self.T_multi-1)/T0)/math.log(self.T_multi))
                T_cur = (current_step-self.warmup_epochs)-T0*(self.T_multi**n_i-1)/(self.T_multi-1)
                T_i = T0*self.T_multi**n_i
            new_lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(T_cur/T_i*math.pi))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr