import json
import os
import pickle
import sys
from os.path import join as pjoin

import numpy as np
import orjson
import pandas
import torch
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback

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


def bisect(f, y, a, b, n, device=None):
    a = torch.tensor(a, device=device).reshape(1, 1)
    b = torch.tensor(b, device=device).reshape(1, 1)
    for _ in range(int(n)):
        c = (a + b) / 2

        mask = f(c) < y

        a = torch.where(mask, c, a)
        b = torch.where(mask, b, c)

    return (a + b) / 2


def dump_task_monotonic_funcs(model, monotonic_funcs):
    task_outs = []
    with torch.inference_mode():
        for task_id, num_labels in enumerate(model.num_labels):
            f = model.scales[task_id].forward
            min_latent = bisect(f, 0, -50, 50, 1e3, device=model.device).item()
            max_latent = bisect(
                f, num_labels - 1, -50, 50, 1e3, device=model.device
            ).item()
            xs = torch.linspace(min_latent, max_latent, 100, device=model.device)
            out = f(xs.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
            task_outs.append(
                {
                    "hidden_to_label": pandas.DataFrame(
                        {"x": xs.cpu().numpy(), "score": out}
                    )
                }
            )
    with open(monotonic_funcs, "wb") as f:
        pickle.dump(task_outs, f)


class DumpWriter:
    def __init__(self, out_base, segments=("train", "test")):
        self.out_base = out_base
        self.segments = segments
        self.step = 0
        self.index_data = {seg: [] for seg in segments}

    def start_step_dump(self, step):
        self.reset_current_epoch()
        self.step = step

    def add_info_full(self, segment, **kwargs):
        for k, v in kwargs.items():
            self.current_epoch_data[segment][k] = ("full", v)

    def add_info_chunk(self, segment, **kwargs):
        seg_data = self.current_epoch_data[segment]
        for k, v in kwargs.items():
            if k in seg_data:
                assert seg_data[k][0] == "chunk"
                seg_data[k][1].append(v)
            else:
                seg_data[k] = ("chunk", [v])

    def ensure_path(self, path):
        os.makedirs(pjoin(self.out_base, path), exist_ok=True)

    def finish_step_dump(self, model):
        from bert_ordinal.experimental_regression import (
            BertForMultiMonotonicTransformSequenceRegression,
        )
        from bert_ordinal.ordinal_models.bert import BertForMultiScaleOrdinalRegression

        thresholds_path = None
        if isinstance(
            model,
            (
                BertForMultiScaleOrdinalRegression,
                BertForMultiMonotonicTransformSequenceRegression,
            ),
        ):
            os.makedirs(self.out_base, exist_ok=True)
            thresholds_path = f"thresholds-{self.step}.pkl"
            full_thresholds_path = pjoin(self.out_base, thresholds_path)
            if isinstance(model, BertForMultiScaleOrdinalRegression):
                dump_task_thresholds(model, full_thresholds_path)
            else:
                dump_task_monotonic_funcs(model, full_thresholds_path)

        for seg in self.segments:
            dump_path = pjoin(seg, f"step-{self.step}.jsonl")
            self.ensure_path(seg)
            data = self.current_epoch_data[seg]
            with open(pjoin(self.out_base, dump_path), "wb") as f:
                keys = data.keys()
                vals = []
                for typ, v in data.values():
                    if typ == "chunk":
                        v = np.hstack(v)
                    vals.append(v)
                for tpl in zip(*vals):
                    d = dict(zip(keys, tpl))
                    f.write(orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY))
                    f.write(b"\n")
            self.index_data[seg].append(
                {
                    "nick": str(self.step),
                    "dump": dump_path,
                    **({"thresholds": thresholds_path} if thresholds_path else {}),
                }
            )

    def reset_current_epoch(self):
        self.current_epoch_data = {seg: {} for seg in self.segments}

    def finish_dump(self):
        with open(pjoin(self.out_base, "index.json"), "w") as f:
            json.dump(self.index_data, f)


class DumpWriterCallback(TrainerCallback):
    def __init__(self, out_base):
        self.dump_writer = DumpWriter(out_base)

    def on_train_end(self, args, state, control, **kwargs):
        self.dump_writer.finish_dump()
