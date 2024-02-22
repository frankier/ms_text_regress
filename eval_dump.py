import argparse
import json
from pprint import pprint

import pandas

from ms_text_regress.eval import evaluate_pred_dist_avgs
from ms_text_regress.label_dist import PRED_AVGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Input file of an eval dump", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.path
    cols = [*PRED_AVGS, "label", "scale_points"]

    def load_row(d):
        return [d[col] for col in cols]

    with open(path) as f:
        records = [load_row(json.loads(line)) for line in f]
    df = pandas.DataFrame.from_records(records, columns=cols)
    pprint(evaluate_pred_dist_avgs(df, df["label"], df["scale_points"]))


if __name__ == "__main__":
    main()
