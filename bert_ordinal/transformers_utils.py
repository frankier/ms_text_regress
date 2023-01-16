import json
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import join as pjoin
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.models.bert.modeling_bert import BertConfig
from transformers.utils import ModelOutput


class BertMultiLabelsMixin:
    # Overwrite num_labels <=> id2label behaviour
    @property
    def num_labels(self) -> List[int]:
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels: List[int]):
        self._num_labels = num_labels


class BertMultiLabelsConfig(BertMultiLabelsMixin, BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def inference_run(
    model,
    tokenizer,
    train_dataset,
    batch_size,
    sample_size=None,
    train_mode=False,
    eval_mode=False,
    use_tqdm=False,
    pass_task_ids=False,
):
    if sample_size:
        train_dataset = train_dataset.shuffle(seed=42).select(range(sample_size))
    dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
        pin_memory=True,
    )

    bar = tqdm(total=len(dataloader), disable=not use_tqdm)
    with torch.inference_mode():
        if eval_mode or train_mode:
            orig_training = model.training
            if train_mode:
                model.train()
            else:
                model.eval()
        for batch in dataloader:
            if pass_task_ids:
                kwargs = {"task_ids": batch["task_ids"]}
            else:
                kwargs = {}
            result = model.forward(
                input_ids=batch["input_ids"].to(model.device, non_blocking=True),
                attention_mask=batch["attention_mask"].to(
                    model.device, non_blocking=True
                ),
                token_type_ids=batch["token_type_ids"].to(
                    model.device, non_blocking=True
                ),
                **kwargs,
            )
            yield batch, result
            bar.update(1)
        if train_mode or eval_mode:
            model.train(orig_training)


class NormalizeHiddenMixin:
    def init_std_hidden_pilot(self, train_dataset, tokenizer, sample_size, batch_size):
        hiddens = torch.vstack(
            [
                out.hidden_linear
                for out in inference_run(
                    self, tokenizer, train_dataset, batch_size, sample_size
                )
            ]
        )
        self.init_std_hidden(hiddens)

    def init_std_hidden(self, hiddens):
        mean = hiddens.mean()
        std = hiddens.std()
        self.classifier.bias.data = (self.classifier.bias.data - mean) / std
        self.classifier.weight.data /= std


def group_labels(task_ids: np.array, labels: np.array):
    sort_idxs = task_ids.argsort()
    task_ids = task_ids[sort_idxs]
    labels = labels[sort_idxs]
    grouped_task_ids, task_group_idxs = np.unique(task_ids, return_index=True)
    groups = np.split(labels, task_group_idxs[1:])
    return grouped_task_ids, groups


@dataclass
class LatentRegressionOutput(ModelOutput):
    """
    Base class for outputs of models with a final hidden linear scale such as
    ordinal regression models and multi-scale linear regression.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Ordinal regression loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`, *optional*, returned when `task_ids` is provided):
            Ordinal regression (or regression) scores (before SoftMax).
        hidden_linear (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Latent variable, before logits is applied.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_linear: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def auto_load(model):
    with open(pjoin(model, "config.json"), "rb") as f:
        config = json.load(f)
    architecture = config["architectures"][0]
    if architecture == "BertForMultiScaleOrdinalRegression":
        from bert_ordinal import BertForMultiScaleOrdinalRegression

        return BertForMultiScaleOrdinalRegression.from_pretrained(model)
    elif architecture == "BertForMultiScaleSequenceClassification":
        from bert_ordinal.baseline_models.classification import (
            BertForMultiScaleSequenceClassification,
        )

        return BertForMultiScaleSequenceClassification.from_pretrained(model)
    elif architecture == "BertForMultiScaleSequenceRegression":
        from bert_ordinal.baseline_models.regression import (
            BertForMultiScaleSequenceRegression,
        )

        return BertForMultiScaleSequenceRegression.from_pretrained(model)
    elif architecture == "BertForMultiMonotonicTransformSequenceRegression":
        from bert_ordinal.experimental_regression import (
            BertForMultiMonotonicTransformSequenceRegression,
        )

        return BertForMultiMonotonicTransformSequenceRegression.from_pretrained(model)
    else:
        raise ValueError(f"Unknown architecture {architecture}")


def auto_pipeline(model, *args, **kwargs):
    from bert_ordinal.baseline_models.classification import (
        BertForMultiScaleSequenceClassification,
    )
    from bert_ordinal.baseline_models.regression import (
        BertForMultiScaleSequenceRegression,
    )
    from bert_ordinal.pipelines import (
        OrdinalRegressionPipeline,
        TextClassificationPipeline,
        TextRegressionPipeline,
    )

    if "device" not in kwargs and torch.cuda.is_available():
        kwargs["device"] = 0

    if isinstance(model, BertForMultiScaleSequenceClassification):
        return TextClassificationPipeline(model=model, *args, **kwargs)
    elif isinstance(model, BertForMultiScaleSequenceRegression):
        return TextRegressionPipeline(model=model, *args, **kwargs)
    else:
        return OrdinalRegressionPipeline(model=model, *args, **kwargs)


@contextmanager
def silence_warnings():
    from transformers import logging

    logging.set_verbosity_error()
    yield
    logging.set_verbosity_warning()
