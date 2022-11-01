"""
This module tests the models, but note that they are really more like smoke
tests, just to test that training and evaluation can run on a single example.
"""
import os

from transformers import TrainingArguments

from bert_ordinal import (
    BertForMultiCutoffOrdinalRegression,
    BertForOrdinalRegression,
    Trainer,
)
from datasets import load_dataset

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "fixtures",
)


def tokenize_dataset(dataset):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

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
            optim="adamw_torch",
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
    )


def test_bert_for_ordinal_regression(tmp_path):
    dataset = tokenize_dataset(
        load_dataset(
            "json", data_files=os.path.join(FIXTURE_DIR, "tiny_shoe_reviews.json")
        )["train"]
    )
    num_labels = 5

    model = BertForOrdinalRegression.from_pretrained(
        "bert-base-cased", num_labels=num_labels
    )
    trained = mk_toy_trainer(tmp_path, model, dataset).train()
    assert trained.global_step == 1


def test_bert_for_multi_ordinal_regression(tmp_path):
    dataset = tokenize_dataset(
        load_dataset(
            "json",
            data_files=os.path.join(FIXTURE_DIR, "tiny_cross_domain_reviews.json"),
        )["train"]
    )
    num_labels = [5, 10]

    model = BertForMultiCutoffOrdinalRegression.from_pretrained(
        "bert-base-cased", num_labels=num_labels
    )
    trained = mk_toy_trainer(tmp_path, model, dataset).train()
    assert trained.global_step == 1
