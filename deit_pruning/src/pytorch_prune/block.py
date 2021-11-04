from torch.nn.utils import prune
from torch.nn.utils.prune import (
  _validate_pruning_amount_init, 
  _validate_structured_pruning, 
  _compute_nparams_toprune, 
  _validate_pruning_amount
)
import torch


class BlockPruningMethod(prune.BasePruningMethod):
  # Well pytorch thinks that structured pruning shall have a 'dim' attr
  # and unstructured pruning shall accept 1-d tensor
  # Block pruning satisfies neither of these conditions, so I set PRUNING_TYPE to this value to avoid misused 
  PRUNING_TYPE = "block"

  def __init__(self, amount, block_row, block_col, n='fro'):
    _validate_pruning_amount_init(amount)
    self.amount = amount
    self.block_row = block_row
    self.block_col = block_col
    self.ord = n

  def get_block_view(self, matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    assert rows % self.block_row == 0
    assert cols % self.block_col == 0

    brows = rows // self.block_row
    bcols = cols // self.block_col
    bcnt = brows * bcols
    
    def subview(idx):
        rstart = idx // bcols * self.block_row
        rend = (idx // bcols + 1) * self.block_row
        cstart = idx % bcols * self.block_col
        cend = (idx % bcols + 1) * self.block_col
        
        return matrix[rstart:rend, cstart:cend]
    
    blocks = [subview(idx) for idx in range(bcnt)]
    
    return torch.stack(blocks)

  def compute_mask(self, t, default_mask):
    # _validate_structured_pruning(t)
    assert len(t.shape) == 2
    rows = t.shape[0]
    cols = t.shape[1]
    assert rows % self.block_row == 0
    assert cols % self.block_col == 0
    brows = rows // self.block_row
    bcols = cols // self.block_col
    bcnt = brows * bcols

    nparams_toprune = _compute_nparams_toprune(self.amount, bcnt)
    _validate_pruning_amount(nparams_toprune, bcnt)

    mask = default_mask.clone()
    if nparams_toprune != 0:
      block_view = self.get_block_view(t)
      norms = torch.linalg.norm(block_view, ord=self.ord, dim=(1, 2))
      indices = torch.topk(norms, k=nparams_toprune, largest=False).indices
      this_mask = torch.ones((brows, bcols))
      this_mask.view(-1)[indices] = 0
      this_mask = torch.repeat_interleave(this_mask, self.block_row, dim=0)
      this_mask = torch.repeat_interleave(this_mask, self.block_col, dim=1)
      mask[this_mask == 0] = 0
    return mask


def block_pruning(module, name, amount, block_row, block_col, n='fro'):
  BlockPruningMethod.apply(module, name, amount=amount, block_row=block_row, block_col=block_col, n=n)
  return module
