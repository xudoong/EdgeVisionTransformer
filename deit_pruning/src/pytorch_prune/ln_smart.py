from torch.nn.utils import prune
from torch.nn.utils.prune import (
  _validate_pruning_amount_init, 
  _validate_structured_pruning, 
  _compute_nparams_toprune, 
  _validate_pruning_amount
)
import torch


class LnSmartStructured(prune.BasePruningMethod):
  PRUNING_TYPE = "1d"

  def __init__(self, amount, n=1):
    _validate_pruning_amount_init(amount)
    self.amount = amount
    self.ord = n
    if n != 1:
      print("WARN: LnSmartStructured is only verified in norm ord=1!")

  def make_mask(self, t, dim, indices):
    # Modified frin pytorch LnStructured.make_mask
    # init mask to 1
    mask = torch.ones_like(t)
    # e.g.: slc = [None, None, None], if len(t.shape) = 3
    slc = [slice(None)] * len(t.shape)
    # replace a None at position=dim with indices
    # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
    slc[dim] = indices
    # use slc to slice mask and replace all its entries with 0s
    # e.g.: mask[:, :, [0, 2, 3]] = 0
    mask[slc] = 0
    return mask

  def compute_mask(self, t, default_mask):
    # _validate_structured_pruning(t)
    assert len(t.shape) == 2
    rows = t.shape[0]
    cols = t.shape[1]

    # 1. Calculate whether to prune row (dim=0) or col (dim=1)
    prune_row = False
    test_nparams_toprune = _compute_nparams_toprune(self.amount, min(rows, cols))
    _validate_pruning_amount(test_nparams_toprune, min(rows, cols))
    row_norm_sum = torch.topk(torch.linalg.norm(t, dim=1, ord=self.ord), k=test_nparams_toprune, largest=False).values.sum() / (cols ** (1 / self.ord))  # Is it right to avoid bias between the two norm values?
    col_norm_sum = torch.topk(torch.linalg.norm(t, dim=0, ord=self.ord), k=test_nparams_toprune, largest=False).values.sum() / (rows ** (1 / self.ord))
    # print(row_norm_sum, col_norm_sum)
    if col_norm_sum >= row_norm_sum:
      prune_row = True

    # 2. Prune (actually)
    bcnt = rows if prune_row else cols
    nparams_toprune = _compute_nparams_toprune(self.amount, bcnt)
    _validate_pruning_amount(nparams_toprune, bcnt)

    mask = default_mask.clone()
    if nparams_toprune != 0:
      indices = torch.topk(torch.linalg.norm(t, dim=1 if prune_row else 0), k=nparams_toprune, largest=False).indices
      mask[self.make_mask(t, 0 if prune_row else 1, indices).to(dtype=mask.dtype) == 0] = 0
      
    return mask


def ln_smart_structured(module, name, amount, n=1):
  LnSmartStructured.apply(module, name, amount=amount, n=n)
  return module
