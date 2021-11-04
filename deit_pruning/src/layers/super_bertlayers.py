import transformers
from transformers import BertModel, BertLayer
from transformers.models.bert.modeling_bert import BertEncoder, BertSelfAttention,  BertAttention,BertIntermediate,BertEmbeddings,BertPooler,BertOutput,BertSelfOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import ACT2FN
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
class VA_BertIntermediate(BertIntermediate):
    def __init__(self, config,layerconfig):
        super().__init__(config)
        print(layerconfig['intermediate_size'])
        self.dense = nn.Linear(config.hidden_size, layerconfig['intermediate_size'])
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
class VA_BertOutput(BertOutput):
     def __init__(self, config,layerconfig):
        super().__init__(config)
        self.dense = nn.Linear(layerconfig['intermediate_size'], config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
class VA_BertSelfAttention(BertSelfAttention):
     def __init__(self, config,heads_num):
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = heads_num
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) ##original head size
        self.all_head_size = heads_num * self.attention_head_size
        print('here',heads_num,self.all_head_size)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
class VA_BertSelfOutput(BertSelfOutput):
    def __init__(self, config,head_num):
        super().__init__(config)
        attention_head_size = int(config.hidden_size / config.num_attention_heads) ##original head size
        all_head_size = head_num * attention_head_size
        self.dense = nn.Linear(all_head_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

 
class VA_BertAttention(BertAttention):
    def __init__(self, config,layerconfig):
        super().__init__(config)
        self.self = VA_BertSelfAttention(config,layerconfig['heads'])
        self.output = VA_BertSelfOutput(config,layerconfig['heads'])
        self.pruned_heads = set()

   
  
class VA_BertLayer(BertLayer):
    def __init__(self, config,layerconfig):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VA_BertAttention(config,layerconfig)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = VA_BertAttention(config,layerconfig)
        self.intermediate = VA_BertIntermediate(config,layerconfig)
        self.output = VA_BertOutput(config,layerconfig)

class VA_BertEncoder(BertEncoder):
     def __init__(self, config):
        super().__init__(config)
        self.config = config
        print(self.config)
        self.layer = nn.ModuleList([VA_BertLayer(config,config.layers[str(i)]) for i in range(config.num_hidden_layers)])

class VA_BertModel(BertModel):

     def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = VA_BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
    