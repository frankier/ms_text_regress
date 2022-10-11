import argparse
import datasets
import numpy as np
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, BertForSequenceClassification
)
from bert_ordinal import BertForOrdinalRegression, ordinal_decode_labels_pt
import torch
import os
from sklearn.metrics import cohen_kappa_score
import evaluate


metric_accuracy = evaluate.load("accuracy")
metric_mae = evaluate.load("mae")
metric_mse = evaluate.load("mse")

# All tokenization is done in advance, before any forking, so this seems to be
# safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate, default: %(default)f")
    parser.add_argument("--epochs", default=3.0, type=float, help="Training epochs, default: %(default)d")
    parser.add_argument("--dataset", default="cross_domain_reviews", help="The dataset to use")
    parser.add_argument("--num-samples", type=int, help="how far to subsample the datset")
    parser.add_argument("--threads", type=int, help="number of cpu threads to set")
    parser.add_argument("--use-ipex", action="store_true", default=False)
    parser.add_argument("--use-deepspeed", action="store_true", default=False)
    parser.add_argument("--classification-baseline", action="store_true", default=False)
    return parser.parse_args()


def dec_label(example):
    return {"label": example["label"] - 1}


def load_data(name):
    if name == "shoe_reviews":
        num_labels = 5
        dataset = datasets.load_dataset("juliensimon/amazon-shoe-reviews")
        dataset = dataset.rename_column("labels", "label")
    elif name == "cross_domain_reviews":
        dataset = datasets.load_dataset("frankier/cross_domain_reviews")
        dataset = dataset.rename_column("rating", "label")
        dataset = dataset.map(dec_label)
        num_labels = 10
    else:
        raise RuntimeError("Unknown dataset")
    labels_arr = np.arange(num_labels)
    return dataset, num_labels, labels_arr


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset, num_labels, labels_arr = load_data(args.dataset)
    dataset = (
        dataset.map(tokenize, batched=True)
    )
    if args.num_samples is not None:
        for label in ("train", "test"):
            dataset[label] = dataset[label].shuffle(seed=42).select(range(args.num_samples))
    models = []
    models.append(
        (
            BertForOrdinalRegression.from_pretrained("bert-base-cased", num_labels=num_labels),
            ordinal_decode_labels_pt
        )
    )

    if args.classification_baseline:
        models.append((
            BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels),
            lambda logits: logits.argmax(dim=-1)
        ))

    training_args = TrainingArguments(
        output_dir="train_out",
        logging_strategy="epoch",
        warmup_ratio=0.1,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        optim="adamw_torch",
        use_ipex=args.use_ipex,
        deepspeed=args.use_deepspeed
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        print()
        print("Computing metrics based upon")
        print("labels", labels)
        print("predictions", predictions)
        mse = metric_mse.compute(predictions=predictions, references=labels)
        return {
            **metric_accuracy.compute(predictions=predictions, references=labels),
            **metric_mae.compute(predictions=predictions, references=labels),
            **mse,
            "rmse": (mse["mse"]) ** 0.5,
            "qwk": cohen_kappa_score(predictions, labels, labels=labels_arr, weights="quadratic"),
        }

    for model, preprocess_logits in models:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=lambda logits, _labels: preprocess_logits(logits)
        )
        trainer.train()


if __name__ == "__main__":
    main()
