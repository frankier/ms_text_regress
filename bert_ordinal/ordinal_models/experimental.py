from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertPreTrainedModel,
    SequenceClassifierOutput,
)

from bert_ordinal.initialisation import iter_task_normal_cutoffs
from bert_ordinal.transformers_utils import (
    BertMultiLabelsConfig,
    LatentRegressionOutput,
    NormalizeHiddenMixin,
)


class BertForWithLatentAndSoftMax(BertPreTrainedModel):
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
        self.into_latent = nn.Linear(config.hidden_size, 1)
        self.classifiers = torch.nn.ModuleList(
            [nn.Linear(1, nl) for nl in config.num_labels]
        )

        # Initialize weights and apply final processing
        self.post_init()

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
        hidden_linear = self.into_latent(pooled_output)
        logits_pad = None
        if task_ids is not None:
            # XXX: Need loss function + argmax implemented for nested_tensor to remove this
            logits_pad = torch.nn.utils.rnn.pad_sequence(
                [
                    self.classifiers[task_id](out).squeeze(0)
                    for task_id, out in zip(task_ids, hidden_linear)
                ],
                batch_first=True,
                padding_value=-100,
            )

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

        return LatentRegressionOutput(
            loss=loss,
            logits=logits_pad,
            hidden_linear=hidden_linear,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def onesided_l2_penalty(z):
    return torch.square(torch.clamp(z, 0.0))


def threshold_preds(thresholds, task_ids, hidden_linear):
    preds = torch.empty(
        (len(hidden_linear),), dtype=torch.long, device=hidden_linear.device
    )
    for idx, (task_id, lin) in enumerate(zip(task_ids, hidden_linear)):
        preds[idx] = torch.count_nonzero((lin - thresholds[task_id]) < 0.0)
    return preds


def threshold_loss(thresholds, task_ids, hidden_linear, labels):
    penalties = torch.zeros((), device=hidden_linear.device)
    for task_id, lin, true_label in zip(task_ids, hidden_linear, labels):
        task_thresholds = thresholds[task_id]
        full_thresholds = torch.concat(
            (
                torch.tensor([float("-inf")], device=task_thresholds.device),
                task_thresholds,
                torch.tensor([float("inf")], device=task_thresholds.device),
            )
        )
        penalty = (
            onesided_l2_penalty(full_thresholds[true_label] - lin)
            + onesided_l2_penalty(lin - full_thresholds[true_label + 1])
        ).squeeze()
        penalties += penalty
    return penalties / len(task_ids)


class BertForMultiScaleThresholdRegression(BertPreTrainedModel, NormalizeHiddenMixin):
    config_class = BertMultiLabelsConfig

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("num_labels", torch.tensor(config.num_labels))
        self.num_labels: torch.Tensor
        self.config = config

        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.batch_norm = nn.BatchNorm1d(1, affine=False)
        self.thresholds = nn.ParameterList(
            [torch.zeros(nl - 1) for nl in config.num_labels]
        )

        # Initialize weights and apply final processing
        self.post_init()

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
    ) -> Union[Tuple[torch.Tensor], LatentRegressionOutput]:
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
        hidden_linear = self.classifier(pooled_output)
        hidden_linear = self.batch_norm(hidden_linear)

        preds = threshold_preds(self.thresholds, task_ids, hidden_linear)

        loss = None
        if labels is not None:
            if task_ids is None:
                raise ValueError(
                    "task_ids must be provided if labels are provided"
                    " -- cannot calculate loss without a task"
                )
            loss = threshold_loss(self.thresholds, task_ids, hidden_linear, labels)
        if not return_dict:
            output = (hidden_linear, preds) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LatentRegressionOutput(
            loss=loss,
            hidden_linear=hidden_linear,
            logits=preds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pilot_quantile_init(self, train_dataset, sample_size, batch_size):
        for task_id, cutoffs in iter_task_normal_cutoffs(train_dataset):
            self.thresholds[task_id].data.copy_(cutoffs)


class BertForMultiScaleFixedThresholdRegression(BertPreTrainedModel):
    """
    This model is quite similar to `BertForMultiScaleSequenceRegression`, but
    uses fixed thresholds instead of learning them as the threshold and ordinal
    regression models do. The loss is similar to
    `BertForMultiScaleThresholdRegression`.
    """

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
        self.scales = torch.nn.ModuleList([nn.Linear(1, 1) for nl in config.num_labels])
        self.register_buffer("num_labels", torch.tensor(config.num_labels))
        self.thresholds = nn.ParameterList(
            [torch.zeros(nl - 1, requires_grad=False) for nl in config.num_labels]
        )

        # Initialize weights and apply final processing
        self.post_init()

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
            Labels for computing the regression loss. An regression loss (MSE loss) is always used.
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
        hidden_linear = self.classifier(pooled_output)

        scaled_outputs = None
        preds = None
        if task_ids is not None:
            scaled_outputs = torch.vstack(
                [
                    self.scales[task_id](hidden)
                    for task_id, hidden in zip(task_ids, hidden_linear)
                ]
            )
            preds = threshold_preds(self.thresholds, task_ids, scaled_outputs)

        loss = None
        if labels is not None:
            if task_ids is None:
                raise ValueError(
                    "task_ids must be provided if labels are provided"
                    " -- cannot calculate loss without a task"
                )
            loss = threshold_loss(self.thresholds, task_ids, scaled_outputs, labels)
        if not return_dict:
            output = (
                hidden_linear,
                preds,
            ) + outputs[2:]
            return ((loss,) + outputs) if loss is not None else output

        return LatentRegressionOutput(
            loss=loss,
            logits=preds,
            hidden_linear=hidden_linear,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def quantile_init(self, train_dataset):
        for task_id, cutoffs in iter_task_normal_cutoffs(train_dataset):
            self.thresholds[task_id].data.copy_(cutoffs)
            self.thresholds[task_id].data.requires_grad = False
            self.thresholds[task_id].requires_grad = False
