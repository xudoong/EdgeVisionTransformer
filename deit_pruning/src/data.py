import torch
import random

# The training data has ~300GB, loading directly into mem is almost impossible
# That's why we need an iterabledataset here
class AdIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, input_file, rids=False):
      super(AdIterableDataset).__init__()
      self.file = input_file

      # Transformers have already implemented IterableDatasetShard
      # So we don't need to handle it ourselves,
      # or the actual steps taken may be smaller than max_steps

      # if distributed:
      #   self.rank = torch.distributed.get_rank()
      #   self.world_size = torch.distributed.get_world_size()
      # else:
      #   self.rank = 0
      #   self.world_size = 1

      self.rids = rids

    def __iter__(self):
      # worker_info = torch.utils.data.get_worker_info()
      # num_workers = 1 if worker_info is None else worker_info.num_workers
      # local_worker_id = 0 if worker_info is None else worker_info.id
      
      # skip = self.world_size * num_workers
      # idx = self.rank * num_workers + local_worker_id
      # worker_id = self.rank * num_workers + local_worker_id

      with open(self.file, "r", encoding='utf-8') as reader:
        for entry in reader:
          # if idx % skip == worker_id:

          line = entry.rstrip("\n").split("\t")
          labels = torch.tensor([int(line[0])], dtype=torch.long) if self.rids else torch.tensor([float(line[0])], dtype=torch.float)
          input_ids = torch.tensor(list(map(int, line[1].split(" "))), dtype=torch.long)
          train_data = {'labels': labels, 'input_ids': input_ids}
          # idx = idx + 1
          yield train_data
          
          # else:
          #   idx = idx + 1
          #   continue

# work like transformers' tokenizer?
def get_token_att_ids(zero, one, token_ids, type_count=2):
    attention_mask = torch.min(token_ids, one)
    token_type_ids = torch.nn.functional.pad(torch.cumsum( torch.where(token_ids[:,0:-1] == 102, one, zero), dim=1), pad=(1,0), mode='constant', value=0)
    if type_count == 2:
        token_type_ids = torch.min(torch.mul(attention_mask, token_type_ids), one)
    else:
        token_type_ids = torch.min(torch.mul(attention_mask, token_type_ids), zero)
    return attention_mask, token_type_ids

# For small dataset (validset)
class AdDataset(torch.utils.data.Dataset):
  def __init__(self, input_file, model_structure="EarlyCrossModel", distributed=False, rids=False):
    assert model_structure == "EarlyCrossModel"
    assert distributed is False  # hasn't test distribute training yet
    # self.zero = torch.nn.parameter.Parameter(torch.tensor(0), requires_grad=False)
    # self.one = torch.nn.parameter.Parameter(torch.tensor(1), requires_grad=False)
    self.data = []
    with open(input_file, "r", encoding='utf-8') as reader:
      for entry in reader:
        line = entry.rstrip("\n").split("\t")
        labels = torch.tensor([int(line[0])], dtype=torch.long) if rids else torch.tensor([float(line[0])], dtype=torch.float)
        # labels = torch.tensor([int(line[0]), 1-int(line[0])], dtype=torch.long) if rids else torch.tensor([float(line[0]), 1-float(line[0])], dtype=torch.float)
        input_ids = torch.tensor(list(map(int, line[1].split(" "))), dtype=torch.long)
        # attention_mask, token_type_ids = get_token_att_ids(self.zero, self.one, input_ids.unsqueeze(0)) # TODO: optimize
        # train_data = [labels, input_ids, attention_mask[0], token_type_ids[0]]
        train_data = [labels, input_ids]
        self.data.append(train_data)
  
  def __getitem__(self, idx):
    # labels, input_ids, attention_mask, token_type_ids = self.data[idx]
    labels, input_ids = self.data[idx]

    return {
      'input_ids': input_ids,
      # 'token_type_ids': token_type_ids,
      # 'attention_mask': attention_mask,
      'labels': labels
    }

  def __len__(self):
    return len(self.data)

# Shuffling IterableDataset
class ShuffleDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset, buffer_size):
    super().__init__()
    self.dataset = dataset
    self.buffer_size = buffer_size

  def set_epoch(self, seed):
    random.seed(seed)
      
  def __iter__(self):
    shufbuf = []
    try:
      dataset_iter = iter(self.dataset)
      for i in range(self.buffer_size):
        shufbuf.append(next(dataset_iter))
    except:
      self.buffer_size = len(shufbuf)
        
    try:
      while True:
        try:
          item = next(dataset_iter)
          evict_idx = random.randint(0, self.buffer_size - 1)
          yield shufbuf[evict_idx]
          shufbuf[evict_idx] = item
        except StopIteration:
          if len(shufbuf) > 0:
            yield shufbuf.pop()
          else:
            break
    except GeneratorExit:
      pass

# get large dataset line count
# saves memory
def _make_gen(reader):
  size = 1024 * 1024
  b = reader(size)
  while b:
    yield b
    b = reader(size)

def rawgencount(filename):
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)
