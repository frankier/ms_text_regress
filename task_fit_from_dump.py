import argparse
import json
from pprint import pprint

import orjson
import pandas

from ms_text_regress.label_dist import PRED_AVGS


def load_data(path, zip_with=None, zip_with_seg=None):
    # Copypasted from viewer.py
    if zip_with is not None:
        import datasets

        zip_with_data = datasets.load_from_disk(zip_with)[zip_with_seg]
        with open(path, "rb") as f:
            records = [
                {**orjson.loads(line), **rec}
                for line, rec in zip(f, zip_with_data, strict=True)
            ]
    else:
        with open(path, "rb") as f:
            records = [orjson.loads(line) for line in f]
    df = pandas.DataFrame(
        {
            k: row[k]
            for k in [
                "label",
                "scale_points",
                "critic_name",
                "task_ids",
                "pred/refit/linear",
            ]
        }
        for row in records
    )
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Input file of an eval dump", required=True)
    return parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Input file of an eval dump")
    parser.add_argument("--zip-with")
    parser.add_argument("--zip-with-seg")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading data")
    df = load_data(args.path, args.zip_with, args.zip_with_seg)
    print("Loaded")

    critics = []
    groups = df.groupby("task_ids")
    idx = 1
    for task_id, grp_df in groups:
        print(f"{idx}/{len(groups)}")
        pprint(grp_df.iloc[0])
        range = grp_df.iloc[0]["scale_points"] - 1
        critic_name = grp_df.iloc[0]["critic_name"]
        ms_mae = (grp_df["pred/refit/linear"] - grp_df["label"]).abs().mean() / range
        critics.append((ms_mae, critic_name))
        idx += 1
    critics.sort()
    pprint(critics)


if __name__ == "__main__":
    main()
