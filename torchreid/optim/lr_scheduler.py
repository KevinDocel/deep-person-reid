from __future__ import print_function, absolute_import

import torch
from .warmupmultistep_lr_scheduler import WarmupMultiStepLR
from .exponentialdecay_lr_scheduler import ExponentialDecayLR

AVAI_SCH = ['single_step', 'multi_step', 'cosine', 'warmup_multi_step', 'exponential_decay']


def build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=1,
        gamma=0.1,
        max_epoch=1,
        warmup_factor=0.01,
        warmup_epoch=10,
        warmup_method='linear'
):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        stepsize (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``stepsize`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``stepsize`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', stepsize=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', stepsize=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError(
            'Unsupported scheduler: {}. Must be one of {}'.format(
                lr_scheduler, AVAI_SCH
            )
        )

    if lr_scheduler == 'single_step':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(stepsize, list):
            raise TypeError(
                'For multi_step lr_scheduler, stepsize must '
                'be a list, but got {}'.format(type(stepsize))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    elif lr_scheduler == 'warmup_multi_step':
        scheduler = WarmupMultiStepLR(
            optimizer=optimizer,
            milestones=stepsize,
            gamma=gamma,
            warmup_factor=warmup_factor,
            warmup_iters=warmup_epoch,
            warmup_method=warmup_method,
        )
    elif lr_scheduler == 'exponential_decay':
        if isinstance(stepsize, list):
            stepsize = stepsize[-1]

        if not isinstance(stepsize, int):
            raise TypeError(
                'For exponential_decay lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
            )

        scheduler = ExponentialDecayLR(
            optimizer=optimizer,
            max_epoch=max_epoch,
            start_decay_at_epoch=stepsize,
            gamma=gamma
        )

    return scheduler
