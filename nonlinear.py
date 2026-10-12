import argparse

import orjson
import pandas
import statsmodels.formula.api as smf

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
        {k: row[k] for k in ["label", "scale_points", "task_ids", "hidden"]}
        for row in records
    )
    return df


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
    pvalues = []
    # XXX: Why are there 266 when there are 798 at training time?
    # Factor of 3? Why? Multiple tasks started
    groups = df.groupby("task_ids")
    idx = 1
    for task_id, grp_df in groups:
        print(f"{idx}/{len(groups)}")
        results = smf.ols("label ~ hidden + I(hidden**2)", data=grp_df).fit()
        pvalues.append(results.pvalues[2])
        idx += 1
    print(pvalues)
    print(sum((1 for p in pvalues if p <= 0.05)))
    print(sum((1 for p in pvalues if p <= 0.05)) / len(pvalues))
    print(sum((1 for p in pvalues if p <= 0.01)))
    print(sum((1 for p in pvalues if p <= 0.01)) / len(pvalues))
    print(sum((1 for p in pvalues if p <= 0.05 / len(pvalues))))
    print(sum((1 for p in pvalues if p <= 0.05 / len(pvalues))) / len(pvalues))
    print(sum((1 for p in pvalues if p <= 0.01 / len(pvalues))))
    print(sum((1 for p in pvalues if p <= 0.01 / len(pvalues))) / len(pvalues))


if __name__ == "__main__":
    main()
