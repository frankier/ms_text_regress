import argparse
import datasets
import numpy as np
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer
)
from bert_ordinal import BertForOrdinalRegression, ordinal_decode_labels_np
import torch
import os
from sklearn.metrics import cohen_kappa_score


# All tokenization is done in advance, before any forking, so this seems to be
# safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-3, type=float, help="LR Default %(default)f")
    parser.add_argument("--steps", default=500, type=int, help="Total training steps %(default)d")
    parser.add_argument("--dataset", default="cross_domain_reviews", help="The dataset to use")
    parser.add_argument("--num-samples", type=int, help="how far to subsample the datset")
    parser.add_argument("--threads", type=int, help="number of cpu threads to set")
    parser.add_argument("--use-ipex", action="set_true", default=False)
    return parser.parse_args()


def load_data(name):
    if name == "shoe_reviews":
        num_labels = 5
        dataset = datasets.load_dataset("juliensimon/amazon-shoe-reviews")
        dataset = dataset.rename_column("labels", "label")
        dataset = dataset.map(
            lambda examples: {"label": examples["label"] + 1}
        )
    elif name == "cross_domain_reviews":
        dataset = datasets.load_dataset("frankier/cross_domain_reviews")
        dataset = dataset.rename_column("rating", "label")
        num_labels = 10
    else:
        raise RuntimeError("Unknown dataset")
    labels_arr = np.arange(1, num_labels + 1)
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
    model = BertForOrdinalRegression.from_pretrained("bert-base-cased", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="epoch",
        optim="adamw_torch",
        use_ipex=args.use_ipex
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = ordinal_decode_labels_np(logits)
        print("logits", logits)
        print("labels", labels)
        print("predictions", predictions)
        print("labels", labels_arr)
        return {
            "qwk": cohen_kappa_score(predictions, labels, labels=labels_arr, weights="quadratic")
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    main()
