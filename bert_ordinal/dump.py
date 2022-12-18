import json
import pickle
import sys

import pandas
import torch
from transformers import AutoTokenizer

from bert_ordinal.datasets import auto_dataset
from bert_ordinal.transformers_utils import auto_pipeline

LOGIT_99 = torch.logit(torch.tensor(0.99))


def dump_results(model, dataset, out, head=None, ds_split="test"):
    dataset, num_labels = auto_dataset(dataset)
    if num_labels != model.config.num_labels:
        print("Warning: num_labels mismatch", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    pipeline = auto_pipeline(model=model, tokenizer=tokenizer)
    with open(out, "w") as f:
        for idx, row in enumerate(dataset[ds_split]):
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
            min_latent = (offsets - abs(LOGIT_99 / discrimination)).min()
            max_latent = (offsets + abs(LOGIT_99 / discrimination)).max()
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
            task_outs.append(
                {
                    "hidden_to_elmo": task_info_long,
                    "discrimination": discrimination,
                    "offsets": offsets,
                }
            )
    with open(task_thresholds, "wb") as f:
        pickle.dump(task_outs, f)
