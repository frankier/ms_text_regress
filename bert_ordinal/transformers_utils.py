import json
from os.path import join as pjoin
from typing import List


class BertMultiLabelsMixin:
    # Overwrite num_labels <=> id2label behaviour
    @property
    def num_labels(self) -> List[int]:
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels: List[int]):
        self._num_labels = num_labels


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
    else:
        raise ValueError(f"Unknown architecture {architecture}")
