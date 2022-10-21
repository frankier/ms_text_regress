import argparse
import numpy as np
from transformers import (
    AutoTokenizer, TrainingArguments, BertForSequenceClassification
)
from bert_ordinal import (
    BertForOrdinalRegression,
    Trainer,
    ordinal_decode_labels_pt
)
from bert_ordinal.datasets import load_data
from bert_ordinal.eval import qwk, qwk_multi_norm
import torch
import os
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


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset, num_labels, is_multi = load_data(args.dataset)
    dataset = dataset.map(tokenize, batched=True)
    if args.num_samples is not None:
        for label in ("train", "test"):
            dataset[label] = dataset[label].shuffle(seed=42).select(range(args.num_samples))
    models = []
    if is_multi:
        import packaging.version
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.13"):
            print(
                f"Warning: multi-scale datasets such as {args.dataset} are not support with torch < 1.13",
                file=sys.stderr
            )
        from bert_ordinal import ordinal_decode_multi_labels_pt, BertForMultiCutoffOrdinalRegression
        models.append((
            BertForMultiCutoffOrdinalRegression.from_pretrained("bert-base-cased", num_labels=num_labels),
            lambda logits: ordinal_decode_multi_labels_pt(logits[2])
        ))
    else:
        models.append((
            BertForOrdinalRegression.from_pretrained("bert-base-cased", num_labels=num_labels),
            ordinal_decode_labels_pt
        ))

    if args.classification_baseline:
        if is_multi:
            raise RuntimeError("There is no classification_baseline for multi-task datasets yet")
        models.append((
            BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels),
            lambda logits: logits.argmax(dim=-1)
        ))

    if is_multi:
        label_names = ["labels", "task_ids"]
    else:
        label_names = None

    training_args = TrainingArguments(
        output_dir="train_out",
        logging_strategy="epoch",
        warmup_ratio=0.1,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        label_names=label_names,
        optim="adamw_torch",
        use_ipex=args.use_ipex,
        deepspeed=args.use_deepspeed
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if is_multi:
            labels, task_ids = labels
            batch_num_labels = np.empty(len(task_ids), dtype=np.int32)
            for idx, task_id in enumerate(task_ids):
                batch_num_labels[idx] = num_labels[task_id]
        else:
            batch_num_labels = num_labels
            
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
        print(f" ** Training model {model.__class__.__name__} on dataset {args.dataset} ** ")
        print("")
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
