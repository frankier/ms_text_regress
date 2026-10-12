"""
This module tests the models, but note that they are really more like smoke
tests, just to test that training and evaluation can run on a single example.
"""

import os
from itertools import repeat

import pytest
from datasets import load_dataset
from transformers import TrainingArguments

from ms_text_regress import (
    BertForMultiScaleOrdinalRegression,
    BertForOrdinalRegression,
    Trainer,
)
from ms_text_regress.baseline_models.classification import (
    BertForMultiScaleSequenceClassification,
)
from ms_text_regress.element_link import DEFAULT_LINK_NAME
from ms_text_regress.ordinal_models.bert import DEFAULT_MULTI_LABEL_DISCRIMINATION_MODE

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "fixtures",
)


BASE = "bert-base-cased"
TOKENIZER = BASE
# BASE = "prajjwal1/bert-tiny"
# TOKENIZER = "bert-base-uncased"


def tokenize_dataset(dataset):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    return dataset.map(tokenize, batched=True)


def mk_toy_trainer(tmp_path, model, dataset):
    return Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=tmp_path,
            logging_strategy="no",
            learning_rate=1e-4,
            lr_scheduler_type="linear",
            warmup_ratio=0.5,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            optim="adamw_torch",
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
    )


LINKS = [
    "fwd_cumulative",
    "bwd_cumulative",
    "fwd_sratio",
    "bwd_sratio",
    "fwd_cratio",
    "bwd_cratio",
    "fwd_acat",
    "bwd_acat",
]

MULTI_LABEL_DISCRIMINATION_MODES = [
    "none",
    "single",
    "per_task",
    "multi",
]


@pytest.mark.parametrize("link", LINKS)
def test_bert_for_ordinal_regression(tmp_path, link):
    dataset = tokenize_dataset(
        load_dataset(
            "json", data_files=os.path.join(FIXTURE_DIR, "tiny_shoe_reviews.json")
        )["train"]
    )
    num_labels = 5

    model = BertForOrdinalRegression.from_pretrained(
        BASE, num_labels=num_labels, link=link
    )
    trained = mk_toy_trainer(tmp_path, model, dataset).train()
    assert trained.global_step == 1


@pytest.fixture
def multiscale_dataset():
    dataset = tokenize_dataset(
        load_dataset(
            "json",
            data_files=os.path.join(FIXTURE_DIR, "tiny_cross_domain_reviews.json"),
        )["train"]
    )
    num_labels = [5, 10]
    return dataset, num_labels


@pytest.mark.parametrize(
    "link,discrimination_mode",
    {
        *zip(LINKS, repeat(DEFAULT_MULTI_LABEL_DISCRIMINATION_MODE)),
        *zip(repeat(DEFAULT_LINK_NAME), MULTI_LABEL_DISCRIMINATION_MODES),
    },
)
def test_bert_for_multi_ordinal_regression(
    tmp_path, link, discrimination_mode, multiscale_dataset
):
    dataset, num_labels = multiscale_dataset
    model = BertForMultiScaleOrdinalRegression.from_pretrained(
        BASE, num_labels=num_labels, link=link
    )
    trained = mk_toy_trainer(tmp_path, model, dataset).train()
    assert trained.global_step == 1


def test_bert_for_multi_scale_classification(tmp_path, multiscale_dataset):
    dataset, num_labels = multiscale_dataset
    model = BertForMultiScaleSequenceClassification.from_pretrained(
        BASE, num_labels=num_labels
    )
    trained = mk_toy_trainer(tmp_path, model, dataset).train()
    assert trained.global_step == 1
