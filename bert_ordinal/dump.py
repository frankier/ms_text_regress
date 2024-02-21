import json
import os
import pickle
import sys
from os.path import join as pjoin

import orjson
import pandas
import torch
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback

from bert_ordinal.datasets import auto_dataset
from bert_ordinal.ordinal_models.vglm import link_of_family_name
from bert_ordinal.scripts.utils import SPLITS
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


def dump_task_thresholds(model, task_thresholds, link, coefs=None):
    task_outs = []
    if coefs is None:

        def get_params(task_id):
            return model.cutoffs.task_summary(task_id)

        def forward(xs, task_id):
            return (
                torch.vstack(
                    model.cutoffs(
                        xs.to(device=model.device).unsqueeze(-1),
                        torch.tensor(task_id, device=model.device).repeat(100),
                    ).unbind()
                )
                .sigmoid()
                .cpu()
                .numpy()
            )

    else:

        def get_params(task_id):
            if coefs.get(task_id) is None:
                return None, None
            return (
                torch.tensor(coefs[task_id][0, :]),
                torch.tensor(coefs[task_id][1, :]),
            )

        def forward(xs, task_id):
            intercepts = coefs[task_id][0, :]
            coef = coefs[task_id][1, :]
            logits = xs.unsqueeze(-1) * coef + intercepts
            return logits.sigmoid()  # .numpy()

    with torch.inference_mode():
        for task_id in range(len(model.num_labels)):
            discrimination, offsets = get_params(task_id)
            if discrimination is None:
                task_outs.append({})
                continue
            min_latent = (offsets - abs(LOGIT_99 / discrimination)).min()
            max_latent = (offsets + abs(LOGIT_99 / discrimination)).max()
            xs = torch.linspace(min_latent, max_latent, 100)
            out = forward(xs, task_id)
            # ordinal_logits = model.cutoffs.discrimination[task_id]
            task_info_wide = pandas.DataFrame(
                {"x": xs, **{str(idx): out[:, idx] for idx in range(out.shape[1])}}
            )
            task_info_long = task_info_wide.melt(
                "x", var_name="index", value_name="score"
            )
            task_info_long["index"] = pandas.to_numeric(task_info_long["index"])
            task_info_long["subprob"] = task_info_long["index"].map(
                link.repr_subproblem
            )
            task_outs.append(
                {
                    "num_labels": model.num_labels[task_id],
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


def dump_task_affines(model, task_affines, coefs=None):
    task_outs = []
    if coefs is not None:

        def get_coefs(task_id):
            if coefs.get(task_id, None) is None:
                return {}
            return {
                "weight": coefs[task_id][1].item(),
                "bias": coefs[task_id][0].item(),
            }

    else:

        def get_coefs(task_id):
            return {
                "weight": model.scales[task_id].weight.item(),
                "bias": model.scales[task_id].bias.item(),
            }

    for task_id, num_labels in enumerate(model.num_labels):
        task_outs.append(
            {
                "num_labels": num_labels,
                **get_coefs(task_id),
            }
        )
    with open(task_affines, "wb") as f:
        pickle.dump(task_outs, f)


def is_refittable(model):
        refittables = []
        try:
            from bert_ordinal.baseline_models.regression import (
                BertForMultiScaleSequenceRegression,
            )
            refittables.append(BertForMultiScaleSequenceRegression)
        except Exception:
            pass
        try:
            from bert_ordinal.experimental_regression import (
                BertForMultiMonotonicTransformSequenceRegression,
            )
            refittables.append(BertForMultiMonotonicTransformSequenceRegression)
        except Exception:
            pass
        try:
            from bert_ordinal.ordinal_models.bert import BertForMultiScaleOrdinalRegression
            refittables.append(BertForMultiScaleOrdinalRegression)
        except Exception:
            pass

        return isinstance(model, tuple(refittables))


class DumpWriter:
    def __init__(self, out_base, zip_with=None, segments=SPLITS):
        self.out_base = out_base
        self.segments = segments
        self.step = 0
        self.index_data = {seg: [] for seg in segments}
        if zip_with is not None:
            self.index_data["_zip_with"] = zip_with

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

    def dump_refit_heads(self, name, model, coefs=None):
        if not is_refittable(model):
            return
        os.makedirs(self.out_base, exist_ok=True)
        name_bits = name.split("/")
        name_escaped = name.replace("/", "__")
        thresholds_path = f"{name_escaped}-{self.step}.pkl"
        full_thresholds_path = pjoin(self.out_base, thresholds_path)
        if name_bits[0] == "refit":
            # Super stringly typed & ripe for refactoring
            assert coefs is not None
            refit_type = name_bits[1]
            if refit_type == "linear":
                dump_task_affines(model, full_thresholds_path, coefs=coefs)
            else:
                dump_task_thresholds(
                    model,
                    full_thresholds_path,
                    link_of_family_name(refit_type),
                    coefs=coefs,
                )
        else:
            assert coefs is None
            from bert_ordinal.baseline_models.regression import BertForMultiScaleSequenceRegression
            from bert_ordinal.ordinal_models.bert import BertForMultiScaleOrdinalRegression
            if isinstance(model, BertForMultiScaleOrdinalRegression):
                dump_task_thresholds(model, full_thresholds_path, model.link)
            elif isinstance(model, BertForMultiScaleSequenceRegression):
                dump_task_affines(model, full_thresholds_path)
            else:
                dump_task_monotonic_funcs(model, full_thresholds_path)
        return thresholds_path

    def dump_cur_step_data(self, seg):
        dump_path = pjoin(seg, f"step-{self.step}.jsonl")
        self.ensure_path(seg)
        data = self.current_epoch_data[seg]
        with open(pjoin(self.out_base, dump_path), "wb") as f:
            keys = data.keys()
            vals = []
            for typ, v in data.values():
                if typ == "chunk":
                    v = (y for x in v for y in x)
                vals.append(v)
            for tpl in zip(*vals):
                d = dict(zip(keys, tpl))
                f.write(orjson.dumps(d, option=orjson.OPT_SERIALIZE_NUMPY))
                f.write(b"\n")
        return dump_path

    def add_heads(self, name, model, coefs=None):
        thresholds_path = self.dump_refit_heads(name, model, coefs=coefs)
        self.current_epoch_data.setdefault("_heads", {})[name] = thresholds_path

    def finish_step_dump(self):
        for seg in self.segments:
            dump_path = self.dump_cur_step_data(seg)
            self.index_data[seg].append(
                {
                    "nick": str(self.step),
                    "dump": dump_path,
                }
            )

    def reset_current_epoch(self):
        self.current_epoch_data = {seg: {} for seg in self.segments}

    def finish_dump(self):
        from itertools import pairwise

        same_length = self.segments[:]
        if "_heads" in self.index_data:
            same_length.append("_heads")
        for seg1, seg2 in pairwise(same_length):
            if len(self.index_data[seg1]) != len(self.index_data[seg2]):
                print(
                    f"WARNING! Different number of steps in segments {seg1} and {seg2}",
                    file=sys.stderr,
                )
        with open(pjoin(self.out_base, "index.json"), "w") as f:
            json.dump(self.index_data, f)


class DumpWriterCallback(TrainerCallback):
    def __init__(self, out_base, **kwargs):
        self.dump_writer = DumpWriter(out_base, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True

    def on_train_end(self, args, state, control, **kwargs):
        self.dump_writer.finish_dump()
