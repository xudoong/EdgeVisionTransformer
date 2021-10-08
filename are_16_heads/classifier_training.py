from tqdm import tqdm, trange
from itertools import islice
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from logger import logger


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


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
    n_epochs=None,
    local_rank=-1,
    n_steps=None,
    fp16=False,
    mask_heads_grad=None,
    eval_mode=False,
    num_workers=0,
):
    # arg n_steps: num steps per gpu
    # arg n_steps_per_epoch: num steps per epoch per gpu
    if not (n_steps or n_epochs):
        Warning('Train: both n_steps and n_epochs are None, set n_epochs=1.')
        n_epochs=1
        
    """Train for a fixed number of steps/epochs"""
    model.train()
    # Device
    device = device or next(model.parameters()).device
    # Prepare data loader
    if local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    is_main = local_rank == -1 or local_rank == 0
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=train_batch_size,
        num_workers=num_workers
    )
    # Number of training steps
    n_steps_per_epoch = int(np.ceil(
        len(train_dataloader)
        / gradient_accumulation_steps
    ))
    # Decide the number of steps based on the requested number of epochs
    # (or vice versa)
    if n_steps is None:
        n_steps = n_steps_per_epoch * n_epochs
    else:
        n_epochs = int(np.ceil(n_steps / n_steps_per_epoch))
    # Print stuff
    if verbose and is_main:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_data)}")
        logger.info(f"  Batch size = {train_batch_size}")
        logger.info(f"  Num epochs = {n_epochs}")
        logger.info(f"  Num steps = {n_steps}")
        logger.info(f"  Num steps per epoch = {n_steps_per_epoch}")
    n_remaining_steps = n_steps
    tr_loss = nb_tr_steps = 0
    # Iterate over epochs
    for _ in trange(int(n_epochs), desc="Epoch", disable=disable_progress_bar):
        # Check whether we are doing the full epoch or not
        full_epoch = n_remaining_steps >= n_steps_per_epoch
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
        n_remaining_steps -= n_steps_per_epoch
        if verbose and is_main:
            logger.info(f"Epoch loss = {epoch_tr_loss / epoch_nb_tr_steps}")
    # Print some info
    if verbose and is_main:
        logger.info("***** Finished training *****")
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
        desc="Training",
        disable=disable_progress_bar,
        total=n_steps,
    )
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_iterator):
        batch = tuple(t.to(device) for t in batch)
        images, labels = batch
        logits = model(images).logits
     
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits, labels) 

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        if fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        # Track loss
        tr_loss += loss.item()
        nb_tr_examples += images.size(0)
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



def get_training_args(
    learning_rate,
    micro_batch_size,
    n_steps,
    n_epochs,
    local_rank,
    num_workers,
    output_dir,
):
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=n_steps or -1,
        num_train_epochs=n_epochs or -1,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        max_grad_norm=1,
        logging_dir='./training_log',
        logging_steps=10,
        local_rank=local_rank,
        save_strategy='no',
        dataloader_num_workers=num_workers,
        ddp_find_unused_parameters=False
    )
    return training_args


def huggingface_trainer_train(
    args,
    train_dataset,
    model,
):
    from transformers import Trainer, TrainingArguments
    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset
    )
    trainer.train()