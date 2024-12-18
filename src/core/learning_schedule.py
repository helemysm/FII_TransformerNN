'''

'''

import torch.optim.lr_scheduler as lr_scheduler

"""
Custom learning rate scheduler that reduces the learning rate based on a plateau in the loss.
This tracks the loss over a specified patience interval and reduces the learning rate when no improvement is observed.

"""
class StepLRWithLoss:
    def __init__(self, optimizer, step_size, gamma, patience):
        self.optimizer = optimizer
        self.gamma = gamma
        self.patience = patience
        self._rate = optimizer.param_groups[0]['lr']
        self.losses = []

    def step(self, loss):
        self.losses.append(loss)
        if len(self.losses) > self.patience and self.losses[-1] >= min(self.losses[-self.patience:]):
            self._rate *= self.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self._rate
            print(f"Reduced learning rate to {self._rate}")
            self.losses = []  # reset losses
        self.optimizer.step()
        
        
        
