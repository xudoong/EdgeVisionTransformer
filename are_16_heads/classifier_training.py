from tqdm import tqdm, trange
from itertools import islice
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from logger import logger
from pytorch_pretrained_bert.optimization import BertAdam


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def prepare_bert_adam(
    optimizer_grouped_parameters,
    learning_rate,
    num_train_steps,
    warmup_proportion,
    loss_scale=0,
    local_rank=-1,
    fp16=False,
    sgd=False,
):
    """Set up the Adam variant for BERT and the learning rate scheduler"""
    # Prepare optimizer
    t_total = num_train_steps
    # Distributed gimmicks
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    # FP16 gimmicks
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use distributed and fp16 training."
            )

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
    else:
        if sgd:
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=learning_rate,
            )
        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                warmup=warmup_proportion,
                t_total=t_total
            )
    # LR schedule

    def lr_schedule(global_step):
        scale = warmup_linear(global_step / t_total, warmup_proportion)
        return learning_rate * scale

    return optimizer, lr_schedule


def train(
    train_data,
    model,
    optimizer,
    train_batch_size,
    gradient_accumulation_steps=1,
    device=None,
    verbose=False,
    disable_progress_bar=False,
    n_gpu=0,
    global_step=0,
    lr_schedule=None,
    n_epochs=1,
    local_rank=-1,
    n_steps=None,
    fp16=False,
    mask_heads_grad=None,
    eval_mode=False,
):
    """Train for a fixed number of steps/epochs"""
    # Device
    device = device or next(model.parameters()).device
    # Prepare data loader
    if local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=train_batch_size
    )
    # Number of training steps
    n_steps_per_epochs = int(
        len(train_data)
        / train_batch_size
        / gradient_accumulation_steps
    )
    # Decide the number of steps based on the requested number of epochs
    # (or vice versa)
    if n_steps is None:
        n_steps = n_steps_per_epochs * n_epochs
    else:
        n_epochs = int(np.ceil(n_steps / n_steps_per_epochs))
    # Print stuff
    if verbose:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_data)}")
        logger.info(f"  Batch size = {train_batch_size}")
        logger.info(f"  Num steps = {n_steps}")
    n_remaining_steps = n_steps
    tr_loss = nb_tr_steps = 0
    # Iterate over epochs
    for _ in trange(int(n_epochs), desc="Epoch", disable=disable_progress_bar):
        # Check whether we are doing the full epoch or not
        full_epoch = n_remaining_steps >= n_steps_per_epochs
        # Run the epoch
        global_step, epoch_tr_loss, epoch_nb_tr_steps = train_epoch(
            train_dataloader,
            model,
            optimizer,
            train_batch_size,
            global_step=global_step,
            lr_schedule=lr_schedule,
            device=device,
            disable_progress_bar=disable_progress_bar,
            n_gpu=n_gpu,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            n_steps=n_remaining_steps if not full_epoch else None,
            mask_heads_grad=mask_heads_grad,
            eval_mode=eval_mode,
        )
        # Update total loss / nb of steps
        tr_loss += epoch_tr_loss
        nb_tr_steps += epoch_nb_tr_steps
        # Total number of remaining steps
        n_remaining_steps -= n_steps_per_epochs
    # Print some info
    if verbose:
        logger.info("***** Finished training *****")
        logger.info(f"  Global step = {len(train_data)}")
        logger.info(f"  Training loss = {tr_loss/nb_tr_steps:.3f}")
    # Return global step and stuff
    return global_step, tr_loss, nb_tr_steps


def train_epoch(
    train_dataloader,
    model,
    optimizer,
    train_batch_size,
    global_step=0,
    lr_schedule=None,
    device=None,
    disable_progress_bar=False,
    n_gpu=0,
    gradient_accumulation_steps=1,
    fp16=False,
    n_steps=None,
    mask_heads_grad=None,
    eval_mode=False,
):
    """Train for one epoch (or a fixed number of steps)"""
    # Device
    device = device or next(model.parameters()).device
    # Training mode let's go
    if eval_mode:
        model.eval()
    else:
        model.train()
    # Iterator
    if n_steps is not None:
        train_dataloader = islice(train_dataloader, n_steps)
    train_iterator = tqdm(
        train_dataloader,
        desc="Iteration",
        disable=disable_progress_bar,
        total=n_steps,
    )
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_iterator):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        # Track loss
        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            # LR scheduling
            if lr_schedule is not None:
                # modify learning rate with special warm up BERT uses
                lr_this_step = lr_schedule(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            if mask_heads_grad is not None:
                model.bert.mask_heads_grad(mask_heads_grad)
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
    return global_step, tr_loss, nb_tr_steps
