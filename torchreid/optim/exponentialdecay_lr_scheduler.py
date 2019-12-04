from torch.optim.lr_scheduler import _LRScheduler


class ExponentialDecayLR(_LRScheduler):
    def __init__(self, optimizer, max_epoch, start_decay_at_epoch=0, gamma=0.001, last_epoch=-1):
        self.max_epoch = max_epoch
        self.start_decay_at_epoch = start_decay_at_epoch
        self.gamma = gamma
        super(ExponentialDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_decay_at_epoch:
            return self.base_lrs
        else:
            exp = (self.last_epoch - self.start_decay_at_epoch + 1) / (self.max_epoch - self.start_decay_at_epoch + 1)
            return [base_lr * self.gamma ** exp for base_lr in self.base_lrs]
