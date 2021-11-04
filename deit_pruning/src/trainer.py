from dataclasses import dataclass
from transformers import Trainer
import torch.nn as nn
import torch
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from nn_pruning.sparse_trainer import SparseTrainer
from data import get_token_att_ids
from utils import get_distil_loss


@dataclass
class DistilTrainingArguments:
    teacher_model: torch.nn.Module
    distil_temperature: float
    alpha_distil: float 


class TrainerWithTokenizer(Trainer):
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        self.zero = torch.nn.parameter.Parameter(
            torch.tensor(0), requires_grad=False)
        self.one = torch.nn.parameter.Parameter(
            torch.tensor(1), requires_grad=False)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        attention_mask, token_type_ids = get_token_att_ids(
            self.zero, self.one, inputs['input_ids'])
       # print('train with tokenizer',inputs['input_ids'])
        inputs['attention_mask'] = attention_mask
        inputs['token_type_ids'] = token_type_ids

        return super().training_step(model, inputs)

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                        prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None,) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        attention_mask, token_type_ids = get_token_att_ids(
            self.zero, self.one, inputs['input_ids'])

        inputs['attention_mask'] = attention_mask
        inputs['token_type_ids'] = token_type_ids
    #  print('predict with tokenizer',inputs['input_ids'])

        return super().prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)


class SparseWithoutTeacherTrainer(SparseTrainer, Trainer):
    def __init__(self, sparse_args, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the default loss in SparseTrainer because it throws an 
        error when run without distillation
        """
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        self.metrics["ce_loss"] += float(loss.mean())
        self.loss_counter += 1
        return (loss, outputs) if return_outputs else loss


class SparserWithTeacherTrainer(SparseTrainer, Trainer):
    def __init__(self, sparse_args, distil_args: DistilTrainingArguments, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)
        self.teacher_model = distil_args.teacher_model
        self.alpha_distil = distil_args.alpha_distil
        self.distil_temperature = distil_args.distil_temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
           teacher_logits = self.teacher_model(**inputs).logits
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        self.metrics['ce_loss'] += float(loss.mean())
        distil_loss = get_distil_loss(outputs.logits, teacher_logits, self.distil_temperature, 'kldiv')
        self.metrics['distil_loss'] += float(distil_loss)
        loss = (1 - self.alpha_distil) * loss + self.alpha_distil * distil_loss
        self.loss_counter += 1

        return (loss, outputs) if return_outputs else loss


class TrainerWithTeacher(Trainer):
    def __init__(self, distil_args: DistilTrainingArguments, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        self.teacher_model = distil_args.teacher_model
        self.alpha_distil = distil_args.alpha_distil
        self.distil_temperature = distil_args.distil_temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
           teacher_logits = self.teacher_model(**inputs).logits
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        distil_loss = get_distil_loss(outputs.logits, teacher_logits, self.distil_temperature, 'kldiv')
        loss = (1 - self.alpha_distil) * loss + self.alpha_distil * distil_loss

        return (loss, outputs) if return_outputs else loss