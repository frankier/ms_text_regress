import argparse
import json
import os
from os.path import join as pjoin

from torch.multiprocessing import Pool

from bert_ordinal.dump import dump_results, dump_task_thresholds
from bert_ordinal.transformers_utils import auto_load


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="cross_domain_reviews", help="The dataset to use"
    )
    parser.add_argument(
        "--checkpoints", help="Input directory of checkpoint directory", required=True
    )
    parser.add_argument("--results", help="Output directory for full dump")
    parser.add_argument(
        "--nproc", help="Number of worker processes", type=int, default=1
    )
    return parser.parse_args()


def checkpoint_key(checkpoint):
    checkpoint_bits = checkpoint.split("-")
    if checkpoint_bits[0] == "checkpoint":
        return (1, int(checkpoint_bits[1]))
    else:
        return (0, checkpoint)


def main():
    args = parse_args()
    index_data = {"train": [], "test": []}
    with Pool(args.nproc) as pool:
        for checkpoint in sorted(os.listdir(args.checkpoints), key=checkpoint_key):
            checkpoint_nick = str(checkpoint_key(checkpoint)[1])
            model_path = pjoin(args.checkpoints, checkpoint)

            def rpath(path):
                return pjoin(args.results, path)

            model = auto_load(model_path)
            thresholds_path = "thresholds"
            os.makedirs(rpath(thresholds_path), exist_ok=True)
            thresholds_path = pjoin(thresholds_path, checkpoint + ".pkl")
            dump_task_thresholds(model, rpath(thresholds_path))
            for ds_split in ["train", "test"]:
                split_path = ds_split
                os.makedirs(rpath(split_path), exist_ok=True)
                dump_path = pjoin(split_path, checkpoint + ".jsonl")
                pool.apply_async(
                    dump_results, (model, args.dataset, rpath(dump_path), ds_split)
                )
                index_data[ds_split].append(
                    {
                        "nick": checkpoint_nick,
                        "thresholds": thresholds_path,
                        "dump": dump_path,
                    }
                )
    with open(pjoin(args.results, "index.json"), "w") as f:
        json.dump(index_data, f)


if __name__ == "__main__":
    main()
