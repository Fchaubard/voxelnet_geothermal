import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosine(_LRScheduler):
    """Learning rate scheduler with linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_steps, max_steps, last_epoch=-1, min_lr_ratio=0.1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine)
            lrs.append(lr)
        return lrs
