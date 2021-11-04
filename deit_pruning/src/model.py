from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss


class SwiftBERT(BertPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    # self.num_labels = config.num_labels
    self.num_labels = 1
    self.config = config

    self.bert = BertModel(config)
    # self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # self.relu = nn.ReLU()
    self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    self.init_weights()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    last_hidden_state = outputs[0]
    logits = self.classifier(last_hidden_state[:, 0])
    #print('logits',logits.shape,logits)
    # logits = self.relu(self.classifier(last_hidden_state[:, 0]))

    # pooled_output = outputs[1]

    # pooled_output = self.dropout(pooled_output)
    # logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      # if self.config.problem_type is None:
      #   if self.num_labels == 1:
      #       self.config.problem_type = "regression"
      #   elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
      #       self.config.problem_type = "single_label_classification"
      #   else:
      #       self.config.problem_type = "multi_label_classification"

      # if self.config.problem_type == "regression":
      #   loss_fct = MSELoss()
      #   if self.num_labels == 1:
      #       loss = loss_fct(logits.squeeze(), labels.squeeze())
      #   else:
      #       loss = loss_fct(logits, labels)
      # elif self.config.problem_type == "single_label_classification":
      #   loss_fct = CrossEntropyLoss()
      #   loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      # elif self.config.problem_type == "multi_label_classification":
      #   loss_fct = BCEWithLogitsLoss()
      #   loss = loss_fct(logits, labels)
      loss_fct = BCEWithLogitsLoss()
      loss = loss_fct(logits[:, 0:1], labels[:, 0:1])
    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )

class SwiftBERTOutput(SwiftBERT):
  def __init__(self, config):
    super().__init__(config)
    self.sigmoid = nn.Sigmoid()
    self.one = torch.nn.parameter.Parameter(torch.tensor(1), requires_grad=False)

  def forward(self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ):
    outputs = super().forward(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=torch.min(token_type_ids, self.one),
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      labels=labels,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    return self.sigmoid(outputs.logits[0, 0])
