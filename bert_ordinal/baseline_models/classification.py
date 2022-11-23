from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    SequenceClassifierOutput,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from bert_ordinal.transformers_utils import BertMultiLabelsMixin


class BertMultiLabelsConfig(BertMultiLabelsMixin, BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForMultiScaleSequenceClassification(BertPreTrainedModel):
    config_class = BertMultiLabelsConfig

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("num_labels", torch.tensor(config.num_labels))
        self.num_labels: torch.Tensor
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifiers = torch.nn.ModuleList(
            [nn.Linear(config.hidden_size, nl) for nl in config.num_labels]
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        task_ids (`torch.LongTensor` of shape `(batch_size,)`):
            Task ids for each example. Should be in half-open range `(0,
            len(config.num_labels]`.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification regression loss. Indices
            should be in half-open range `[0, ...,
            config.num_labels[task_id])`. An classification loss (Cross-Entropy loss) is always used.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits_pad = None
        if task_ids is not None:
            logits = torch.nested.as_nested_tensor(
                [
                    self.classifiers[task_id](pooled_output).squeeze(0)
                    for task_id in task_ids
                ]
            )
            # XXX: Need loss function + argmax implemented for nested_tensor to remove this
            logits_pad = logits.to_padded_tensor(-100)

        loss = None
        if labels is not None:
            if task_ids is None or logits_pad is None:
                raise ValueError(
                    "task_ids must be provided if labels are provided"
                    " -- cannot calculate loss without a task"
                )
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_pad, labels)
        if not return_dict:
            output = (logits_pad,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_pad,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
