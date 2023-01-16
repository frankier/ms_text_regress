import copy
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

# from UMNN import MonotonicNN
from zuko.nn import MonotonicMLP

from bert_ordinal.baseline_models.regression import (
    BertForMultiScaleSequenceRegressionConfig,
)
from bert_ordinal.transformers_utils import (
    LatentRegressionOutput,
    NormalizeHiddenMixin,
    group_labels,
    inference_run,
)


class BertForMultiMonotonicTransformSequenceRegression(
    BertPreTrainedModel, NormalizeHiddenMixin
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
        # self.scales = torch.nn.ModuleList([MonotonicNN(1, [50, 50], dev=self.device) for nl in config.num_labels])
        self.scales = torch.nn.ModuleList(
            [MonotonicMLP(1, 1, [50, 50]) for nl in config.num_labels]
        )
        if config.loss == "mse":
            self.loss_fct = nn.MSELoss()
        elif config.loss == "mae":
            self.loss_fct = nn.L1Loss()
        elif config.loss == "adjust_l1":
            from bert_ordinal.vendor.adjust_smooth_l1_loss import AdjustSmoothL1Loss

            self.loss_fct = AdjustSmoothL1Loss(num_features=1, beta=1.0)
        else:
            raise ValueError(f"Unknown loss {config.loss}")

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
            loss = self.loss_fct(scaled_outputs, labels.unsqueeze(-1).float())
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

    def norm_hiddens(self, train_dataset, tokenizer, batch_size, sample_size=None):
        hiddens = torch.vstack(
            [
                out[1].hidden_linear
                for out in inference_run(
                    # Use eval mode because we don't want dropout
                    self,
                    tokenizer,
                    train_dataset,
                    batch_size,
                    sample_size=sample_size,
                    eval_mode=True,
                )
            ]
        )
        # Step 1. Normalize up to classifier
        mean = hiddens.mean()
        std = hiddens.std()
        self.classifier.bias.data = (self.classifier.bias.data - mean) / std
        self.classifier.weight.data /= std
        # Step 2.
        return (hiddens - mean) / std

    def pilot_quantile_init(self, train_dataset, tokenizer, sample_size, batch_size):
        hiddens = self.norm_hiddens(train_dataset, tokenizer, batch_size, sample_size)
        task_ids = np.asarray(train_dataset["task_ids"])
        labels = np.asarray(train_dataset["label"])
        grouped_task_ids, groups = group_labels(task_ids, labels)
        for task_id, group in zip(grouped_task_ids, groups):
            # We don't even pay attention to the fact that these examples are not from the correct group here
            # We're just getting things "close enough" without having to do a full pass over the whole dataset
            scale_outs = self.scales[task_id](hiddens)
            scale_mean = scale_outs.mean()
            scale_std = scale_outs.std()
            group_mean = np.mean(group)
            group_std = np.std(group)
            self.scales[task_id][-1].bias.data = (
                self.scales[task_id][-1].bias.data - scale_mean
            ) / scale_std + group_mean
            self.scales[task_id][-1].weight.data *= group_std / scale_std

    def train_scale(self, task_id, hiddens, labels):
        self.scales[task_id].train()
        opt = LBFGS(self.scales[task_id].parameters())

        best_loss = float("inf")
        best_state_dict = None
        for itr in range(0, 10):

            def loss_closure():
                opt.zero_grad()
                out = self.scales[task_id](hiddens)
                loss_val = self.loss_fct(out, labels)
                loss_val.backward()
                return loss_val

            loss = opt.step(loss_closure)
            if math.isnan(loss):
                break
            if loss < best_loss:
                best_loss = loss
                best_state_dict = copy.deepcopy(self.scales[task_id].state_dict())

        self.scales[task_id].load_state_dict(best_state_dict)

    def pilot_train_init(self, train_dataset, tokenizer, batch_size):
        hiddens = self.norm_hiddens(train_dataset, tokenizer, batch_size).squeeze(-1)
        task_ids = torch.tensor(train_dataset["task_ids"], device=self.device)
        labels = torch.tensor(train_dataset["label"], device=self.device)
        task_id_sort_perm = torch.argsort(task_ids)
        hiddens = hiddens[task_id_sort_perm]
        task_ids = task_ids[task_id_sort_perm]
        labels = labels[task_id_sort_perm]
        group_task_ids, group_sizes = torch.unique(task_ids, return_counts=True)
        group_sizes = tuple(group_sizes)
        for task_id, task_hiddens, task_labels in zip(
            group_task_ids,
            torch.split_with_sizes(hiddens, group_sizes),
            torch.split_with_sizes(labels, group_sizes),
        ):
            task_hiddens = torch.sort(task_hiddens)[0]
            task_labels = torch.sort(task_labels)[0]
            self.train_scale(
                task_id, task_hiddens.unsqueeze(-1), task_labels.unsqueeze(-1)
            )
