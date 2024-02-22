from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.bert.modeling_bert import (
    BERT_INPUTS_DOCSTRING,
    BERT_START_DOCSTRING,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from ms_text_regress.transformers_utils import (
    BertMultiLabelsMixin,
    LatentRegressionOutput,
    LossScalersMixin,
    NormalizeHiddenMixin,
    group_labels,
)


class LossConfigMixin:
    def __init__(self, loss="mse", **kwargs):
        super().__init__(**kwargs)

        self.loss = loss


class BertForMultiScaleSequenceRegressionConfig(
    BertMultiLabelsMixin, LossConfigMixin, BertConfig
):
    pass


class RegressionLossMixin(LossScalersMixin):
    def __init__(self, config):
        super().__init__(config)
        if config.loss == "mse":
            self.loss_fct = nn.MSELoss(reduction="none")
        elif config.loss == "mae":
            self.loss_fct = nn.L1Loss(reduction="none")
        elif config.loss == "adjust_l1":
            from ms_text_regress.vendor.adjust_smooth_l1_loss import AdjustSmoothL1Loss

            self.loss_fct = AdjustSmoothL1Loss(
                num_features=1, beta=1.0, reduction="none"
            )
        else:
            raise ValueError(f"Unknown loss {config.loss}")

    def compute_loss(self, scaled_outputs, task_ids, labels):
        return (
            self.loss_fct(scaled_outputs, labels.unsqueeze(-1).float())
            / self.loss_scalers[task_ids]
        ).mean()


class MultiScaleSequenceRegressionMixin(NormalizeHiddenMixin):
    def forward_pooled(
        self,
        outputs,
        pooled_output: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden_linear = self.classifier(pooled_output)

        scaled_outputs = None
        if task_ids is not None:
            scaled_outputs = torch.vstack(
                [
                    self.scales[task_id](hidden)
                    for task_id, hidden in zip(task_ids, hidden_linear)
                ]
            )

        loss = None
        if labels is not None:
            if task_ids is None:
                raise ValueError(
                    "task_ids must be provided if labels are provided"
                    " -- cannot calculate loss without a task"
                )
            loss = self.compute_loss(scaled_outputs, task_ids, labels)
        if not return_dict:
            output = (
                hidden_linear,
                scaled_outputs,
            ) + outputs[2:]
            return ((loss,) + outputs) if loss is not None else output

        return LatentRegressionOutput(
            loss=loss,
            logits=scaled_outputs,
            hidden_linear=hidden_linear,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _zero_bias(self):
        # At initialisation time BERT usually has interquartile range of about
        # 0.02 - 0.05 and a mean of -1 to 1. If we zero the bias of the final
        # shared linear, this seems to reduce the variance of the mean to a
        # typical range of -0.5 to 0.5.
        self.classifier.bias.data = torch.zeros(1)

    def _init_scale_range(self, task_id):
        scale_points = self.num_labels[task_id]
        scale_length = (scale_points.clone().detach() - 1).float()
        self.scales[task_id].weight.data = scale_length[np.newaxis, np.newaxis]
        self.scales[task_id].bias.data = (scale_length / 2)[np.newaxis]

    def _init_scale_empirical(self, task_id, group):
        self.scales[task_id].weight.data = torch.tensor(
            np.std(group)[np.newaxis, np.newaxis], dtype=torch.float32
        )
        self.scales[task_id].bias.data = torch.tensor(
            np.mean(group)[np.newaxis], dtype=torch.float32
        )

    def init_scales_empirical(
        self, task_ids: np.array, labels: np.array, min_samples_per_group=20
    ):
        self._zero_bias()
        self._init_scale_empirical(task_ids, labels, min_samples_per_group)

    def _init_scales_empirical(
        self, task_ids: np.array, labels: np.array, min_samples_per_group=20
    ):
        grouped_task_ids, groups = group_labels(task_ids, labels)
        for task_id, group in zip(grouped_task_ids, groups):
            if len(group) < min_samples_per_group:
                self._init_scale_range(task_id)
            else:
                self._init_scale_empirical(task_id, group)

    def pilot_quantile_init(
        self,
        train_dataset,
        tokenizer,
        sample_size,
        batch_size,
        min_samples_per_group=20,
    ):
        self.init_std_hidden_pilot(train_dataset, tokenizer, sample_size, batch_size)
        task_ids = np.asarray(train_dataset["task_ids"])
        labels = np.asarray(train_dataset["label"])
        self._init_scales_empirical(task_ids, labels, min_samples_per_group)

    def init_scales_range(self):
        for task_id in range(len(self.num_labels)):
            self._init_scale_range(task_id)


@add_start_docstrings(
    """
    Bert Model transformer with a per-task/scale regression.
    """,
    BERT_START_DOCSTRING,
)
class BertForMultiScaleSequenceRegression(
    MultiScaleSequenceRegressionMixin,
    RegressionLossMixin,
    BertPreTrainedModel,
    NormalizeHiddenMixin,
):
    config_class = BertForMultiScaleSequenceRegressionConfig

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

        # Initialize weights and apply final processing
        self.post_init()

    """
    task_ids (`torch.LongTensor` of shape `(batch_size,)`):
        Task ids for each example. Should be in half-open range `(0,
        len(config.num_labels]`.
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the regression loss. An regression loss (MSE loss) is always used.
    """

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
    ) -> Union[Tuple[torch.Tensor], LatentRegressionOutput]:
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
        return self.forward_pooled(
            outputs, pooled_output, task_ids, labels, return_dict
        )
