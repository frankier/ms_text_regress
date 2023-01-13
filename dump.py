import argparse
import sys

from bert_ordinal.baseline_models.classification import (
    BertForMultiScaleSequenceClassification,
)
from bert_ordinal.baseline_models.regression import BertForMultiScaleSequenceRegression
from bert_ordinal.dump import (
    dump_results,
    dump_task_monotonic_funcs,
    dump_task_thresholds,
)
from bert_ordinal.transformers_utils import auto_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="cross_domain_reviews", help="The dataset to use"
    )
    parser.add_argument("--model", help="Input directory of model dump", required=True)
    parser.add_argument("--results", help="Output file for eval dump")
    parser.add_argument(
        "--task-thresholds", help="Output file for label thresholds dump"
    )
    parser.add_argument(
        "--task-mono-funcs", help="Output file for monotonic thresholds dump"
    )
    parser.add_argument("--head", type=int, help="Only evaluate the first N examples")
    parser.add_argument("--split", help="Dataset split to use", default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading model")
    model = auto_load(args.model)
    print("Loaded")
    if args.task_thresholds is not None and (
        isinstance(model, BertForMultiScaleSequenceClassification)
        or isinstance(model, BertForMultiScaleSequenceRegression)
    ):
        print(
            "Dumping task thresholds not supported for BertForMultiScaleSequenceClassification or BertForMultiScaleSequenceRegression",
            file=sys.stderr,
        )
        sys.exit(-1)
    if args.results is not None:
        dump_results(model, args.dataset, args.results, args.head, args.split)
    if args.task_thresholds is not None:
        dump_task_thresholds(model, args.task_thresholds)
    if args.task_mono_funcs is not None:
        dump_task_monotonic_funcs(model, args.task_mono_funcs)


if __name__ == "__main__":
    main()
