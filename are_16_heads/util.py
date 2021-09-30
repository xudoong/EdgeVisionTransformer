import torch
from torchvision import datasets
from torchvision.datasets.folder import ImageFolder
import torch.distributed as dist
from typing import Dict 
import numpy as np
import torch

def head_entropy(p):
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return - plogp.sum(dim=-1)


def head_pairwise_kl(p):
    # p has shape bsz x nheads x L x L and is normalized in the last
    # dim
    logp = torch.log(p)
    logp[p == 0] = -40
    H_pq = -torch.einsum("bilk,bjlk->bijl", [p, logp])
    H_p = head_entropy(p).unsqueeze(-2)
    KL = H_pq - H_p
    KL.masked_fill_(p.sum(-1).eq(0).unsqueeze(1), 0.0)
    KL.masked_fill_(p.sum(-1).eq(0).unsqueeze(2), 0.0)
    return KL


def attn_disagreement(p):
    # p has shape bsz x nheads x L x L and is normalized in the last
    # dim
    n_heads = p.size(1)
    return torch.einsum("bilk,bjlk->b", [p, p]) / n_heads ** 2


def out_disagreement(out):
    # out has shape bsz x nheads x L x d
    n_heads = out.size(1)
    # Normalize
    out /= torch.sqrt((out ** 2).sum(-1)).unsqueeze(-1) + 1e-20
    cosine = torch.einsum("bild,bjld->b", [out, out])
    return cosine / n_heads ** 2


def print_1d_tensor(tensor):
    print("\t".join(f"{x:.5f}" for x in tensor.cpu().data))


def print_2d_tensor(tensor):
    for row in range(len(tensor)):
        print_1d_tensor(tensor[row])


def none_if_empty(string):
    return string if string != "" else None


def get_vit_encoder(model):
    if hasattr(model, 'vit'):
        return model.vit.encoder
    if hasattr(model, 'module'):
        return model.module.vit.encoder
    else:
        raise RuntimeError('Model neither has attribute "vit" or "module".')

def get_vit_config(model):
    if hasattr(model, 'vit'):
        return model.vit.config 
    if hasattr(model, 'module'):
        return model.module.vit.config
    else:
        raise RuntimeError('Model neither has attribute "vit" or "module".')



'''============================================================
        load data
================================================================='''
class DictImageFolder(datasets.ImageFolder):
    def __init__(self, shuffle, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
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