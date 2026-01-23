import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup and restarts.

    This scheduler combines linear warmup, cosine annealing, and periodic restarts.
    After an initial warmup phase, the learning rate follows a cosine decay schedule
    down to a minimum value, then restarts with a new cycle. Each cycle can be longer
    and/or have a reduced maximum learning rate.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    first_cycle_steps : int
        Number of steps in the first cycle (including warmup).
    cycle_mult : float, optional
        Cycle length magnification after each restart (default: 1.0, i.e., constant cycle length).
    max_lr : float, optional
        Maximum learning rate for the first cycle (default: 0.1).
    min_lr : float, optional
        Minimum learning rate (default: 0.001).
    warmup_steps : int, optional
        Number of steps for linear warmup at the start of each cycle (default: 0).
    gamma : float, optional
        Multiplicative factor to decrease max_lr after each cycle (default: 1.0, i.e., no decay).
    last_epoch : int, optional
        The index of the last epoch (default: -1).

    Notes
    -----
    - Learning rate increases linearly from min_lr to max_lr during warmup_steps.
    - After warmup, learning rate follows a cosine annealing schedule from max_lr to min_lr.
    - At the end of each cycle, the scheduler restarts with updated cycle length and max_lr.
    - Useful for training regimes that benefit from periodic restarts and warmup.

    Example
    -------
    >>> scheduler = CosineAnnealingWarmupRestarts(
    ...     optimizer, first_cycle_steps=100, cycle_mult=2, max_lr=0.01, min_lr=1e-4, warmup_steps=10, gamma=0.5
    ... )
    >>> for epoch in range(300):
    ...     train(...)
    ...     scheduler.step()
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


LR_SCHEDULERS = {
    "CosineAnnealingWarmupRestarts": CosineAnnealingWarmupRestarts,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
}
