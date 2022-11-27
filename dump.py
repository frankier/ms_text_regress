import argparse
import json
import sys

from transformers import AutoTokenizer

from bert_ordinal import BertForMultiScaleOrdinalRegression
from bert_ordinal.datasets import load_data, load_from_disk_with_labels
from bert_ordinal.pipelines import OrdinalRegressionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="cross_domain_reviews", help="The dataset to use"
    )
    parser.add_argument("--model", help="Input directory of model dump", required=True)
    parser.add_argument("--out", help="Output file for eval dump", required=True)
    parser.add_argument("--head", type=int, help="Only evaluate the first N examples")
    return parser.parse_args()


def main():
    args = parse_args()
    model = BertForMultiScaleOrdinalRegression.from_pretrained(args.model)
    try:
        dataset, num_labels = load_data(args.dataset)
    except RuntimeError:
        dataset, num_labels = load_from_disk_with_labels(args.dataset)
    if num_labels != model.config.num_labels:
        print("Warning: num_labels mismatch", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    pipeline = OrdinalRegressionPipeline(model=model, tokenizer=tokenizer)
    with open(args.out, "w") as f:
        for idx, row in enumerate(dataset["test"]):
            if idx >= args.head:
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


if __name__ == "__main__":
    main()
