from typing import Dict
import torch
import numpy as np
import random
from torch.utils.data import dataset
from torchvision import datasets
from torchvision.datasets.folder import ImageFolder
import torch.distributed as dist


def replace_key_in_dict(old_pattern, new_pattern, dic):
    keys = list(dic.keys())
    for key in keys:
        if old_pattern in key:
            old_key = key
            new_key = key.replace(old_pattern, new_pattern)
            dic[new_key] = dic.pop(old_key)


def swift_converter(state_dict):
    # name replace
    replace_key_in_dict("dense_act", "dense", state_dict)
    replace_key_in_dict("lin.", "classifier.", state_dict)

    # vocab size adjustment
    state_dict['bert.embeddings.word_embeddings.weight'] = state_dict['bert.embeddings.word_embeddings.weight'][:30522, :]

    # position_ids
    state_dict['bert.embeddings.position_ids'] = torch.arange(
        512).expand((1, -1))

    if state_dict.get("zero") is not None:
        del state_dict["zero"]
    if state_dict.get("one") is not None:
        del state_dict["one"]


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def unzero_parameters(model, epsilon=0.01, head_only=True):
    # from sparse_xp.py
    # Used to avoid zero gradients when doing final finetune on sparse networks that we want to extend
    # Make sure some parts are not completely zero
    for k, v in model.named_parameters():
        if "bias" in k:
            continue
        if head_only and "attention" not in k:
            continue
        zero_mask = v == 0
        if zero_mask.sum() == 0:
            continue

        with torch.no_grad():
            print("unzero_parameters", k, "sparsity=", float(
                zero_mask.sum() / zero_mask.numel()), zero_mask.shape)
            new_values = torch.randn_like(v)
            new_values *= v.std() * epsilon
            new_values += v.mean()
            new_values *= zero_mask
            v.copy_(v + new_values)
    return model


'''============================================================
        load data
================================================================='''


class DictImageFolder(datasets.ImageFolder):
    def __init__(self, shuffle, *args, **kwargs):
        print('Enter DictImageFolder initial function')
        super().__init__(*args, **kwargs)
        print('Super initial finish.')
        self.shuffle = shuffle
        self.idx_list = np.arange(super().__len__())
        if self.shuffle:
            np.random.shuffle(self.idx_list)

    def __getitem__(self, index: int) -> Dict:
        index = self.idx_list[index]
        item = super().__getitem__(index)
        return dict(
            pixel_values=item[0],
            label=item[1]
        )


def build_dataset(data_path, input_size=224, is_train=False, shuffle=False, return_dict=True):
    def build_transform(input_size):
        from torchvision import transforms
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        t = []
        if input_size > 32:
            size = int((256 / 224) * input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(
            IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    import os
    from torchvision import datasets
    transform = build_transform(input_size)
    root = os.path.join(data_path, 'val' if not is_train else 'train')
    if return_dict:
        dataset = DictImageFolder(shuffle, root, transform=transform)
    else:
        dataset = ImageFolder(root, transform=transform)
    num_classes = 1000
    return dataset, num_classes


def to_data_loader(dataset, batch_size, num_workers):
    import torch
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


'''============================================================
        evaluate
==============================================================='''


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def dist_reduce_scalar(value, op, device):
    value = torch.tensor(value).to(device)
    dist.reduce(value, 0, op)
    return value.item()


def evaluate(
    eval_data,
    model,
    eval_batch_size,
    device=None,
    result=None,
    disable_progress_bar=False,
    distributed=False,
    num_workers=0,
):
    from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
    from tqdm import tqdm
    from itertools import islice
    import time

    """Evaluate the model's accuracy"""
    if distributed:
        eval_sampler = DistributedSampler(eval_data)
    else:
        eval_sampler = SequentialSampler(eval_data)

    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=num_workers)

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

    # reduce when distributed
    if distributed:
        for k, v in result.items():
            v = dist_reduce_scalar(
                value=v, op=dist.ReduceOp.SUM, device=device)
            v /= dist.get_world_size()
            result[k] = v

    return result


def dist_print(is_main, *args, **kwargs):
    if is_main:
        print(*args, **kwargs)


'''============================================================
        knowledge distillation
==============================================================='''


def get_distil_loss(student_logits, teacher_logits, temperature: float, loss_func: str):
    from torch.nn import functional as F
    if loss_func.lower() == 'kldiv':
        input_log_prob = torch.nn.LogSoftmax(dim=-1)(student_logits / temperature)
        target_prob = torch.nn.Softmax(dim=-1)(teacher_logits / temperature)
        loss = F.kl_div(
            input=input_log_prob,
            target=target_prob,
            reduction='batchmean'
        ) * (temperature ** 2)
        return loss

    elif loss_func.lower() == 'mse':
        loss = torch.nn.MSELoss(reduction='mean')
        return loss(teacher_logits, student_logits)
    else:
        raise NotImplementedError(
            f"Current implemented distil loss function: [kldiv, mse]. But your loss_func is {loss_func}")


def show_deit_sparsity(model_path):
    def get_sparsity(weight: torch.Tensor):
        rv = torch.sum(torch.abs(weight) <= 1e-7) / np.prod(weight.shape)
        return round(float(rv) * 100, 2)

    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(model_path)
    layers = model.vit.encoder.layer

    for i in range(len(layers)):
        attention = layers[i].attention.attention
        qkv_weight = [attention.query.weight,
                      attention.key.weight, attention.value.weight]
        qkv_sparsity = [get_sparsity(x) for x in qkv_weight]

        attn_output_weight = layers[i].attention.output.dense.weight
        intermediate_weight = layers[i].intermediate.dense.weight
        intermediate_sparsity = get_sparsity(intermediate_weight)

        output_weight = layers[i].output.dense.weight
        output_sparsity = get_sparsity(output_weight)

        print(f'Layer {i} sparsity: qkv {qkv_weight[0].shape} {qkv_sparsity}, attn_output {attn_output_weight.shape} intermediate {intermediate_weight.shape} {intermediate_sparsity}, output {output_weight.shape} {output_sparsity}')


if __name__ == '__main__':
    import sys
    show_deit_sparsity(sys.argv[1])
