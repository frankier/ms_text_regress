import argparse
import json
import pickle
import sys

import pandas
import torch
from transformers import AutoTokenizer

from bert_ordinal.baseline_models.classification import (
    BertForMultiScaleSequenceClassification,
)
from bert_ordinal.datasets import load_data, load_from_disk_with_labels
from bert_ordinal.pipelines import OrdinalRegressionPipeline, TextClassificationPipeline
from bert_ordinal.transformers_utils import auto_load

LOGIT_99 = torch.logit(torch.tensor(0.99))


def dump_results(model, dataset, out, head):
    try:
        dataset, num_labels = load_data(dataset)
    except RuntimeError:
        dataset, num_labels = load_from_disk_with_labels(dataset)
    if num_labels != model.config.num_labels:
        print("Warning: num_labels mismatch", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    if isinstance(model, BertForMultiScaleSequenceClassification):
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    else:
        pipeline = OrdinalRegressionPipeline(model=model, tokenizer=tokenizer)
    with open(out, "w") as f:
        for idx, row in enumerate(dataset["test"]):
            if head is not None and idx >= head:
                break
            output = pipeline(row)
            json.dump(
                {
                    "review_score": row["review_score"],
                    "label": row["label"],
                    "scale_points": row["scale_points"],
                    "movie_title": row["movie_title"],
                    "task_ids": row["task_ids"],
                    "text": row["text"],
                    **output,
                },
                f,
            )
            f.write("\n")


def dump_task_thresholds(model, task_thresholds):
    task_outs = []
    with torch.inference_mode():
        for task_id in range(len(model.num_labels)):
            discrimination, offsets = model.cutoffs.task_summary(task_id)
            min_latent = (offsets - LOGIT_99 / discrimination).min()
            max_latent = (offsets + LOGIT_99 / discrimination).max()
            xs = torch.linspace(min_latent, max_latent, 100)
            out = (
                torch.vstack(
                    model.cutoffs(
                        xs.unsqueeze(-1), torch.tensor(task_id).repeat(100)
                    ).unbind()
                )
                .sigmoid()
                .numpy()
            )
            # ordinal_logits = model.cutoffs.discrimination[task_id]
            task_info_wide = pandas.DataFrame(
                {"x": xs, **{str(idx): out[:, idx] for idx in range(out.shape[1])}}
            )
            task_info_long = task_info_wide.melt(
                "x", var_name="index", value_name="score"
            )
            task_info_long["index"] = pandas.to_numeric(task_info_long["index"])
            task_info_long["subprob"] = task_info_long["index"].map(
                model.link.repr_subproblem
            )
            task_outs.append(task_info_long)
    with open(task_thresholds, "wb") as f:
        pickle.dump(task_outs, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="cross_domain_reviews", help="The dataset to use"
    )
    parser.add_argument("--model", help="Input directory of model dump", required=True)
    parser.add_argument("--results", help="Output file for eval dump")
    parser.add_argument("--task-thresholds", help="Output file for label model dump")
    parser.add_argument("--head", type=int, help="Only evaluate the first N examples")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading model")
    model = auto_load(args.model)
    print("Loaded")
    if args.results is not None:
        dump_results(model, args.dataset, args.results, args.head)
    if args.task_thresholds is not None:
        dump_task_thresholds(model, args.task_thresholds)


if __name__ == "__main__":
    main()
