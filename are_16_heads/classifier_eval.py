from torchvision.datasets.folder import IMG_EXTENSIONS
from tqdm import tqdm
from itertools import islice
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler

from logger import logger
import util
from util import get_vit_config, get_vit_encoder


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def evaluate(
        eval_data,
        model,
        eval_batch_size,
        save_attention_probs=False,
        print_head_entropy=False,
        device=None,
        result=None,
        verbose=True,
        disable_progress_bar=False,
        scorer=None,
        distributed=False,
        num_workers=0,
):
    """Evaluate the model's accuracy"""
    if distributed:
        eval_sampler = DistributedSampler(eval_data)
    else:
        eval_sampler = SequentialSampler(eval_data)
        
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=num_workers)

    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_data)}")
        logger.info(f"  Batch size = {eval_batch_size}")

    model.eval()
    # Device
    device = device or next(model.parameters()).device

    # Run prediction for full data
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    all_predicitions = []
    all_labels = []
    eval_iterator = tqdm(
        eval_dataloader, desc="Evaluating", disable=disable_progress_bar)
    inference_time = 0
    for images, labels in eval_iterator:
        images = images.to(device)
        labels = labels.to(device)

        inference_time -= time.time()
        with torch.no_grad():
            tmp_eval_loss = model(
                images).logits
            logits = tmp_eval_loss

        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=-1)
        inference_time += time.time()
        labels = labels.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, labels)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        all_predicitions.extend(predictions)
        all_labels.extend(labels)

        nb_eval_examples += images.shape[0]
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = result or {}
    result["eval_loss"] = eval_loss
    result["eval_accuracy"] = eval_accuracy
    result["inference_time"] = inference_time
    # Add task specific score
    if scorer is not None:
        all_predicitions = np.asarray(all_predicitions)
        all_labels = np.asarray(all_labels)
        result[scorer.name] = scorer(all_predicitions, all_labels)

    # reduce when distributed
    if distributed:
        score = torch.tensor(result[scorer.name]).to(device)
        dist.reduce(score, 0, dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            score /= dist.get_world_size()
        result[scorer.name] = score.item()
                    
    return result


def calculate_head_importance(
        model,
        data,
        batch_size,
        device=None,
        normalize_scores_by_layer=True,
        verbose=True,
        disable_progress_bar=False,
        subset_size=1.0,
        distributed=False,
        num_workers=0,
        pruned_heads=None
):
    """Calculate head importance scores"""
    # Disable dropout
    model.train() # TO BE FIXED
    # Device
    device = device or next(model.parameters()).device
    if subset_size <= 1:
        subset_size *= len(data)
    n_prune_steps = int(np.ceil(int(subset_size) / batch_size))
    if verbose and (not distributed or dist.get_rank() == 0):
        logger.info("***** Calculating head importance *****")
        logger.info(f"  Num examples = {len(data)}")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num steps = {n_prune_steps}")
        if distributed:
            logger.info(f'  Distributed mode, world size = {dist.get_world_size()}')
        else:
            logger.info(f'  Not in distributed mode.')
    
    # Prepare data loader
    if not distributed:
        sampler = RandomSampler(data)
    else:
        sampler = DistributedSampler(data)

    dataloader = DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers
    )
    if subset_size < 1:
        dataloader = islice(dataloader, n_prune_steps)
    prune_iterator = tqdm(
        dataloader,
        desc="[Cal-head-importance-iteration]",
        disable=disable_progress_bar,
    )
    # Head importance tensor
    config = get_vit_config(model)
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    seq_len = 197
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    tot_tokens = 0

    for step, batch in enumerate(prune_iterator):
        batch = tuple(t.to(device) for t in batch)
        image, label = batch
        # Compute gradients
        loss = model(image).logits.sum()
        loss.backward()

        for layer in range(n_layers):
            encoder = get_vit_encoder(model)
            self_att = encoder.layer[layer].attention.attention
            ctx = self_att.context_layer_val
            grad_ctx = ctx.grad
            
            # Take the dot
            dot = torch.einsum("bhli,bhli->bhl", [grad_ctx, ctx])
            dot = dot.abs().sum(-1).sum(0).detach()
            
            # map remaining heads after exact pruning to original index
            if dot.shape[0] != n_heads:
                if pruned_heads is None:
                    raise RuntimeError('Must provide pruned_heads when useing exact pruning')
                appended_dot = torch.zeros(n_heads).to(device)
                pruned = sorted(list(pruned_heads[layer]))
                dot_i = 0
                for i in range(n_heads):
                    if i in pruned: continue
                    appended_dot[i] = dot[dot_i]
                    dot_i += 1
                head_importance[layer] += appended_dot
            else:
                head_importance[layer] += dot

        tot_tokens += seq_len

    if distributed:
        dist.all_reduce(head_importance, op=dist.ReduceOp.SUM)
        
        tot_tokens_tensor = torch.tensor(tot_tokens).to(device)
        dist.all_reduce(tot_tokens_tensor, op=dist.ReduceOp.SUM)
        tot_tokens = tot_tokens_tensor.item()

    head_importance[:-1] /= tot_tokens
    head_importance[-1] /= subset_size
    # Layerwise importance normalization
    if normalize_scores_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1/exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    return head_importance


def predict(
    predict_data,
    model,
    predict_batch_size,
    device=None,
    verbose=True,
    disable_progress_bar=False,
):
    """Predict labels on a dataset"""

    predict_sampler = SequentialSampler(predict_data)
    predict_dataloader = DataLoader(
        predict_data, sampler=predict_sampler, batch_size=predict_batch_size)

    if verbose:
        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = {len(predict_data)}")
        logger.info(f"  Batch size = {predict_batch_size}")

    model.eval()
    # Device
    device = device or next(model.parameters()).device

    predict_iterator = tqdm(
        predict_dataloader,
        desc="Predicting labels",
        disable=disable_progress_bar
    )

    # Compute model predictions
    predictions = []
    for input_ids, input_mask, segment_ids, label_ids in predict_iterator:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        # Compute logits
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        # Track predictions
        batch_predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
        for pred in batch_predictions:
            predictions.append(pred)

    return np.asarray(predictions, dtype=int)


def analyze_nli(anal_examples, predictions, labels_list):
    report = {
        "label": {},
        "lex_sem": {},
        "pred_arg_struct": {},
        "logic": {},
        "knowledge": {},
        "domain": {},
    }
    normalizers = {k: {} for k in report}
    for example, pred in zip(anal_examples, predictions):
        correct = float(example.label == labels_list[pred])
        for feature in report:
            values = getattr(example, feature)
            if values is not None:
                # Sometimes there are multiple values
                for value in values.split(";"):
                    # Record whether the model was correct on this particular
                    # value of the feature
                    if value not in report[feature]:
                        report[feature][value] = 0
                        normalizers[feature][value] = 0
                    report[feature][value] += correct
                    normalizers[feature][value] += 1
    # Normalize report
    for feature in report:
        Ns = normalizers[feature]
        report[feature] = {k: v / Ns[k] for k, v in report[feature].items()}

    return report
