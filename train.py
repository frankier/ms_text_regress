import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import evaluate
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
)

from bert_ordinal import BertForOrdinalRegression, Trainer, ordinal_decode_labels_pt
from bert_ordinal.datasets import load_data
from bert_ordinal.eval import qwk, qwk_multi_norm

metric_accuracy = evaluate.load("accuracy")
metric_mae = evaluate.load("mae")
metric_mse = evaluate.load("mse")

# All tokenization is done in advance, before any forking, so this seems to be
# safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ExtraArguments:
    dataset: str
    num_samples: Optional[int] = None
    threads: Optional[int] = None
    classification_baseline: bool = False
    trace_labels_predictions: bool = False
    num_dataset_proc: Optional[int] = None
    warm_dataset_cache: bool = False


_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return _tokenizer


def tokenize(text):
    tokenizer = get_tokenizer()
    return tokenizer(text, padding="max_length", truncation=True, return_tensors="np")


def main():
    parser = HfArgumentParser((TrainingArguments, ExtraArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        training_args, args = parser.parse_args_into_dataclasses()

    # args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dataset, num_labels, is_multi = load_data(
        args.dataset, num_dataset_proc=args.num_dataset_proc
    )
    dataset = dataset.map(
        tokenize,
        input_columns="text",
        batched=True,
        desc="Tokenizing",
        num_proc=args.num_dataset_proc,
    )
    if args.num_samples is not None:
        for label in ("train", "test"):
            dataset[label] = (
                dataset[label].shuffle(seed=42).select(range(args.num_samples))
            )
    if args.warm_dataset_cache:
        print("Dataset cache warmed")
        return

    models = []
    if is_multi:
        import packaging.version

        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.13"):
            print(
                f"Warning: multi-scale datasets such as {args.dataset} are not support with torch < 1.13",
                file=sys.stderr,
            )
        from bert_ordinal import BertForMultiScaleOrdinalRegression

        bert_mcor = BertForMultiScaleOrdinalRegression.from_pretrained(
            "bert-base-cased", num_labels=num_labels
        )
        link = bert_mcor.link

        def proc_multiord_logits(logits):
            return torch.hstack([link.top_from_logits(li) for li in logits[1].unbind()])

        models.append((bert_mcor, proc_multiord_logits))
    else:
        models.append(
            (
                BertForOrdinalRegression.from_pretrained(
                    "bert-base-cased", num_labels=num_labels
                ),
                ordinal_decode_labels_pt,
            )
        )

    if args.classification_baseline:
        if is_multi:
            raise RuntimeError(
                "There is no classification_baseline for multi-task datasets yet"
            )
        models.append(
            (
                BertForSequenceClassification.from_pretrained(
                    "bert-base-cased", num_labels=num_labels
                ),
                lambda logits: logits.argmax(dim=-1),
            )
        )

    if is_multi:
        label_names = ["labels", "task_ids"]
    else:
        label_names = None

    training_args.label_names = label_names
    training_args.optim = "adamw_torch"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if is_multi:
            labels, task_ids = labels
            batch_num_labels = np.empty(len(task_ids), dtype=np.int32)
            for idx, task_id in enumerate(task_ids):
                batch_num_labels[idx] = num_labels[task_id]
        else:
            batch_num_labels = num_labels

        if args.trace_labels_predictions:
            print()
            print("Computing metrics based upon")
            print("labels", labels)
            print("predictions", predictions)

        mse = metric_mse.compute(predictions=predictions, references=labels)
        res = {
            **metric_accuracy.compute(predictions=predictions, references=labels),
            **metric_mae.compute(predictions=predictions, references=labels),
            **mse,
            "rmse": (mse["mse"]) ** 0.5,
        }
        if is_multi:
            res["qwk"] = qwk_multi_norm(predictions, labels, batch_num_labels)
        else:
            res["qwk"] = qwk(predictions, labels, batch_num_labels)
        return res

    for model, preprocess_logits in models:
        print("")
        print(
            f" ** Training model {model.__class__.__name__} on dataset {args.dataset} ** "
        )
        print("")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=lambda logits, _labels: preprocess_logits(
                logits
            ),
        )
        trainer.train()


if __name__ == "__main__":
    main()
