"""
This module contains classes to train/evaluate BERT with ordinal regression
heads based upon and in the style of the HuggingFace Transformers library, in
particular the BertForSequenceClassification class.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy
import packaging.version
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import Trainer as OriginalHFTrainer
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from bert_ordinal.ordinal import (
    OrdinalCutoffs,
    OrdinalRegressionOutput,
    ordinal_encode_labels,
)


@add_start_docstrings(
    """
    Bert Model transformer with an ordinal regression head on top (a linear layer on top of the pooled
    output) e.g. for essay grading.
    """,
    BERT_START_DOCSTRING,
)
class BertForOrdinalRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.cutoffs = OrdinalCutoffs(config.num_labels)

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
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], OrdinalRegressionOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence ordinal regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
        hidden_logits = self.classifier(pooled_output)
        ordinal_logits = self.cutoffs(hidden_logits)

        loss = None

        if labels is not None:
            labels_ord_enc = ordinal_encode_labels(labels, self.num_labels)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(ordinal_logits, labels_ord_enc)
        if not return_dict:
            output = (
                hidden_logits,
                ordinal_logits,
            ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return OrdinalRegressionOutput(
            loss=loss,
            hidden_logits=hidden_logits,
            ordinal_logits=ordinal_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NestedTensorTrainerMixin:
    def _pad_across_processes(self, tensor, *args, **kwargs):
        if getattr(tensor, "is_nested", False):
            return tensor
        return super()._pad_across_processes(tensor, *args, **kwargs)


class Trainer(NestedTensorTrainerMixin, OriginalHFTrainer):
    pass


if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.13"):
    from bert_ordinal.ordinal import (
        MultiOrdinalCutoffs,
        bce_with_logits_ragged_mean,
        ordinal_encode_multi_labels,
    )

    class BertMultiLabelsConfig(BertConfig):
        # Overwrite num_labels <=> id2label behaviour
        @property
        def num_labels(self) -> int:
            return self._num_labels

        @num_labels.setter
        def num_labels(self, num_labels: List[int]):
            self._num_labels = num_labels

    @add_start_docstrings(
        """
        Bert Model transformer with an ordinal regression head on top (a linear layer on top of the pooled
        output) e.g. for essay grading.
        """,
        BERT_START_DOCSTRING,
    )
    class BertForMultiCutoffOrdinalRegression(BertPreTrainedModel):
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
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.cutoffs = MultiOrdinalCutoffs(config.num_labels)

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
        ) -> Union[Tuple[torch.Tensor], OrdinalRegressionOutput]:
            """
            task_ids (`torch.LongTensor` of shape `(batch_size,)`):
                Task ids for each example. Should be in half-open range `(0,
                config.num_labels]`.
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence ordinal regression loss. Indices
                should be in half-open range `[0, ...,
                config.num_labels[task_id])`. An ordinal regression loss --- binary
                cross entropy on the ordinal encoded labels --- is always used.
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
            hidden_logits = self.classifier(pooled_output)

            ordinal_logits = None
            task_cutoffs = None
            if task_ids is not None:
                ordinal_logits, task_cutoffs = self.cutoffs(hidden_logits, task_ids)

            loss = None
            if labels is not None:
                if ordinal_logits is None:
                    raise ValueError(
                        "task_ids must be provided if labels are provided"
                        " -- cannot calculate loss without a task"
                    )
                batch_num_labels = torch.gather(self.num_labels, 0, task_ids)
                labels_ord_enc = ordinal_encode_multi_labels(labels, batch_num_labels)
                loss_fct = BCEWithLogitsLoss(reduction="sum")
                loss = bce_with_logits_ragged_mean(ordinal_logits, labels_ord_enc)
            if not return_dict:
                output = (
                    hidden_logits,
                    task_cutoffs,
                    ordinal_logits,
                ) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return OrdinalRegressionOutput(
                loss=loss,
                hidden_logits=hidden_logits,
                task_cutoffs=task_cutoffs,
                ordinal_logits=ordinal_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
