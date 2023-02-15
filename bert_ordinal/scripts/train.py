import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import join as pjoin
from os.path import relpath
from pprint import pprint
from typing import Optional

import evaluate
import numpy as np
import torch
from mt_ord_bert_utils import get_tokenizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_pt_utils import nested_numpify

from bert_ordinal import Trainer
from bert_ordinal.datasets import load_from_disk_with_labels
from bert_ordinal.dump import DumpWriterCallback
from bert_ordinal.element_link import link_registry
from bert_ordinal.eval import add_bests, evaluate_pred_dist_avgs, evaluate_predictions
from bert_ordinal.label_dist import (
    PRED_AVGS,
    clip_predictions_np,
    summarize_label_dist,
    summarize_label_dists,
)
from bert_ordinal.transformers_utils import inference_run, silence_warnings

metric_accuracy = evaluate.load("accuracy")
metric_mae = evaluate.load("mae")
metric_mse = evaluate.load("mse")

# All tokenization is done in advance, before any forking, so this seems to be
# safe.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ExtraArguments:
    dataset: str
    num_samples: Optional[int] = None
    model: str = None
    discrimination_mode: str = "per_task"
    threads: Optional[int] = None
    trace_labels_predictions: bool = False
    smoke: bool = False
    pilot_quantiles: bool = False
    pilot_train_init: bool = False
    pilot_sample_size: int = 256
    peak_class_prob: float = 0.5
    dump_initial_model: Optional[str] = None
    fitted_ordinal: Optional[str] = None
    dump_results: Optional[str] = None
    sampler: str = "default"
    num_vgam_workers: int = 8
    initial_probe: bool = False
    initial_probe_lr: Optional[float] = None
    initial_probe_steps: Optional[float] = None
    scale_lr_multiplier: Optional[float] = None
    use_bert_large_wholeword: bool = False
    no_loss_scale: bool = False
    compile_model: bool = False


def prepare_dataset_for_fast_inference(dataset, label_names, sort=False):
    wanted_columns = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "label",
        "label_ids",
        "scale_points",
        "length",
        *label_names,
    }
    dataset = dataset.remove_columns(list(set(dataset.column_names) - wanted_columns))
    if sort:
        return dataset.sort("length")
    else:
        return dataset


def init_weights(training_args, args, model_conf, model, tokenizer, dataset):
    if args.pilot_quantiles:
        # We wait until after Trainer is initialised to make sure the model is on the GPU
        if model_conf["is_ordinal"]:
            model.pilot_quantile_init(
                dataset["train"],
                tokenizer,
                args.pilot_sample_size,
                training_args.per_device_train_batch_size,
                peak_class_prob=args.peak_class_prob,
            )
        else:
            model.pilot_quantile_init(
                dataset["train"],
                tokenizer,
                args.pilot_sample_size,
                training_args.per_device_train_batch_size,
            )
    if args.pilot_train_init:
        model.pilot_train_init(
            dataset["train"], tokenizer, training_args.per_device_train_batch_size
        )
    if model_conf["name"] == "fixed_threshold":
        model.quantile_init(dataset["train"])
    if args.fitted_ordinal:
        model.init_std_hidden_pilot(
            dataset["train"],
            tokenizer,
            args.pilot_sample_size,
            training_args.per_device_train_batch_size,
        )
        model.set_ordinal_heads(torch.load(args.fitted_ordinal))
    if args.initial_probe:
        linear_probe_args = TrainingArguments(
            learning_rate=args.initial_probe_lr
            if args.initial_probe_lr is not None
            else training_args.learning_rate,
            max_steps=args.initial_probe_steps or -1,
            num_train_epochs=1.0,
            prediction_loss_only=True,
            output_dir=pjoin(training_args.output_dir, "linear_probe"),
            save_strategy="no",
            lr_scheduler_type="constant",
        )
        linear_probe_trainer = Trainer(
            model=model,
            args=linear_probe_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
        )
        for param in model.bert.parameters():
            param.requires_grad = False
        linear_probe_trainer.train()
        for param in model.bert.parameters():
            param.requires_grad = True
    if not args.no_loss_scale and model_conf["has_continuous_output"]:
        model.init_loss_scaling(dataset["train"])


def get_mono_optimizers(model, training_args, extra_args):
    from transformers.optimization import get_scheduler

    if training_args.weight_decay > 0:
        raise ValueError("Weight decay not supported for monotonic regression")
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
        training_args
    )
    lr = optimizer_kwargs["lr"]
    optimizer = optimizer_cls(
        [
            {"params": model.shared_parameters(), "lr": lr},
            {
                "params": model.scales.parameters(),
                "lr": lr * extra_args.scale_lr_multiplier,
            },
        ],
        **optimizer_kwargs,
    )
    linear_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(training_args.max_steps),
        num_training_steps=training_args.max_steps,
    )
    linear_scheduler.lr_lambdas[0]
    scheduler = LambdaLR(optimizer, [linear_scheduler.lr_lambdas[0], lambda x: 1.0])
    return optimizer, scheduler


def modconf(name, *args, **kwargs):
    is_kwargs = {
        "has_latent",
        "is_mono",
        "is_regress",
        "is_ordinal",
        "has_continuous_output",
        "is_threshold",
    }
    for k in is_kwargs:
        kwargs[k] = k in args

    return (name, {**kwargs, "name": name})


def ensure_context(pool):
    import contextlib

    if pool is None:
        return contextlib.nullcontext()
    else:
        return pool


MODEL_ALIAS_DECODE = dict(
    [
        modconf(
            "regress", "has_latent", "is_regress", "has_continuous_output", loss="mse"
        ),
        modconf(
            "regress_l1",
            "has_latent",
            "is_regress",
            "has_continuous_output",
            loss="mae",
        ),
        modconf(
            "regress_adjust_l1",
            "has_latent",
            "is_regress",
            "has_continuous_output",
            loss="adjust_l1",
        ),
        modconf("mono", "has_latent", "is_mono", "has_continuous_output", loss="mse"),
        modconf(
            "mono_l1", "has_latent", "is_mono", "has_continuous_output", loss="mae"
        ),
        modconf(
            "mono_adjust_l1",
            "has_latent",
            "is_mono",
            "has_continuous_output",
            loss="adjust_l1",
        ),
        *(
            modconf(link, "has_latent", "is_ordinal", link=link)
            for link in link_registry
        ),
        modconf("class"),
        modconf("latent_softmax", "has_latent"),
        modconf("threshold", "has_latent", "is_threshold"),
        modconf("fixed_threshold", "has_latent", "is_threshold"),
        modconf("metric", "has_latent"),
    ]
)


class ClassPredProc:
    @staticmethod
    def proc_logits(logits):
        label_dists = logits.softmax(dim=-1)
        return (
            label_dists,
            *summarize_label_dists(label_dists).values(),
        )

    @staticmethod
    def postprocess(pred_label_dists, batch_num_labels):
        label_dists = pred_label_dists[0]
        agg_preds = dict(zip(PRED_AVGS, pred_label_dists[1:]))
        return {
            "label_dists": label_dists,
            "agg_pred": agg_preds,
        }


class OrdinalPredProc:
    def __init__(self, link):
        self.link = link

    def proc_logits(self, logits):
        label_dists = [
            self.link.label_dist_from_logits(li) for li in logits[0].unbind()
        ]
        return (
            logits[1],
            label_dists,
            *summarize_label_dists(label_dists).values(),
        )

    def postprocess(self, pred_label_dists, batch_num_labels):
        hiddens = pred_label_dists[0]
        label_dists = pred_label_dists[1]
        agg_preds = dict(zip(PRED_AVGS, pred_label_dists[2:]))

        return {
            "label_dists": label_dists,
            "agg_pred": agg_preds,
            "hidden": hiddens,
        }


class LogitsPassthroughMixin:
    @staticmethod
    def proc_logits(logits):
        return logits


class LatentContinuousOutPredProc(LogitsPassthroughMixin):
    @staticmethod
    def postprocess(pred_label_dists, batch_num_labels):
        raw_predictions, hiddens = pred_label_dists
        predictions = clip_predictions_np(raw_predictions, batch_num_labels)
        return {"hidden": hiddens, "pred": predictions}

    @staticmethod
    def dump_callback(batch, result):
        # XXX: This could maybe be replaced by proc_logits/postprocess
        return {
            "pred": clip_predictions_np(
                result.logits.detach().cpu().numpy(), batch.scale_points.cpu().numpy()
            )
        }


def flatten_dump(d):
    res = {}
    for k, v in d.items():
        if k == "agg_pred":
            res.update(v)
        else:
            res[k] = v
    return res


class TrainerAndEvaluator:
    def __init__(self):
        self.training_args, self.args = self.get_args()
        self.config_libs(self.args)
        self.model_conf = MODEL_ALIAS_DECODE[self.args.model]
        self.dataset, self.num_labels = self.load_dataset(self.args)
        self.model, self.label_names, self.pred_proc, self.proc_logits = self.get_model(
            self.model_conf, self.args, self.num_labels
        )
        if self.args.compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model)
        self.eval_dataset = self.prepare_eval_dataset(
            self.args, self.dataset, self.label_names
        )
        self.training_args.label_names = self.label_names
        self.training_args.optim = "adamw_torch"
        self.optimizers = self.get_optimizers(
            self.model_conf, self.model, self.training_args, self.args
        )

    @staticmethod
    def get_args():
        parser = HfArgumentParser((TrainingArguments, ExtraArguments))

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            training_args, args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            training_args, args = parser.parse_args_into_dataclasses()
        return training_args, args

    @staticmethod
    def config_libs(args):
        torch.backends.cuda.matmul.allow_tf32 = True

        if args.threads:
            torch.set_num_threads(args.threads)

    @staticmethod
    def load_dataset(args):
        import packaging.version

        dataset, num_labels = load_from_disk_with_labels(args.dataset)

        if args.num_samples is not None:
            for label in ("train", "test"):
                dataset[label] = (
                    dataset[label].shuffle(seed=42).select(range(args.num_samples))
                )

        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.13"):
            print(
                f"Warning: multi-scale datasets such as {args.dataset} are not support with torch < 1.13",
                file=sys.stderr,
            )

        return dataset, num_labels

    @staticmethod
    def prepare_eval_dataset(args, dataset, label_names):
        eval_train_dataset = prepare_dataset_for_fast_inference(
            dataset["train"], label_names, sort=not args.dump_results
        )
        eval_test_dataset = prepare_dataset_for_fast_inference(
            dataset["test"], label_names, sort=not args.dump_results
        )
        eval_dataset = {"train": eval_train_dataset, "test": eval_test_dataset}
        return eval_dataset

    @staticmethod
    def get_optimizers(model_conf, model, training_args, args):
        if model_conf["is_mono"]:
            return get_mono_optimizers(model, training_args, args)
        else:
            return (None, None)

    @classmethod
    def get_model(cls, model_conf, args, num_labels):
        if args.smoke:
            base_model = "prajjwal1/bert-tiny"
            torch.set_num_threads(1)
        elif args.use_bert_large_wholeword:
            base_model = "bert-large-cased-whole-word-masking"
        else:
            base_model = "bert-base-cased"

        if isinstance(num_labels, int):
            return cls.get_single_scale_model(model_conf, args, num_labels, base_model)
        else:
            return cls.get_multi_scale_model(model_conf, args, num_labels, base_model)

    @staticmethod
    def get_single_scale_model(model_conf, args, num_labels, base_model):
        label_names = ["labels"]
        pred_proc = None
        proc_logits = None
        if model_conf["name"] == "class":
            from transformers import BertForSequenceClassification

            model = BertForSequenceClassification.from_pretrained(
                base_model, num_labels=num_labels
            )

            def proc_logits(logits):
                label_dist = logits.softmax(dim=-1)
                return (label_dist, *summarize_label_dist(label_dist).values())

        elif model_conf["name"] == "regress":
            from transformers import BertForSequenceClassification

            with silence_warnings():
                model = BertForSequenceClassification.from_pretrained(
                    base_model, num_labels=num_labels, problem_type="regression"
                )
            proc_logits = None
        elif model_conf["is_ordinal"]:
            from bert_ordinal import BertForOrdinalRegression

            model = BertForOrdinalRegression.from_pretrained(
                base_model,
                num_labels=num_labels,
                link=args.model,
                discrimination_mode=args.discrimination_mode,
            )
            link = model.link

            def proc_logits(logits):
                label_dist = link.label_dist_from_logits(logits[0])
                return (
                    logits[1],
                    label_dist,
                    *summarize_label_dist(label_dist).values(),
                )

        else:
            print(f"Unknown model type {args.model}", file=sys.stderr)
            sys.exit(-1)
        return model, label_names, pred_proc, proc_logits

    @staticmethod
    def proc_logits_passthrough(logits):
        return logits

    @classmethod
    def get_multi_scale_model(cls, model_conf, args, num_labels, base_model):
        label_names = ["labels", "task_ids"]
        model_kwargs = {
            "pretrained_model_name_or_path": base_model,
            "num_labels": num_labels,
        }
        pred_proc = None
        proc_logits = None
        if model_conf["name"] == "class":
            from bert_ordinal.baseline_models.classification import (
                BertForMultiScaleSequenceClassification,
            )

            model_cls = BertForMultiScaleSequenceClassification
            pred_proc = ClassPredProc
        elif model_conf["has_continuous_output"]:
            if model_conf["is_regress"]:
                from bert_ordinal.baseline_models.regression import (
                    BertForMultiScaleSequenceRegression,
                )

                model_cls = BertForMultiScaleSequenceRegression
                loss_map = {
                    "regress": "mse",
                    "regress_l1": "mae",
                    "regress_adjust_l1": "adjust_l1",
                }
            else:
                from bert_ordinal.experimental_regression import (
                    BertForMultiMonotonicTransformSequenceRegression,
                )

                model_cls = BertForMultiMonotonicTransformSequenceRegression
                loss_map = {
                    "mono": "mse",
                    "mono_l1": "mae",
                    "mono_adjust_l1": "adjust_l1",
                }
            model_kwargs["loss"] = loss_map[args.model]
            pred_proc = LatentContinuousOutPredProc
        elif model_conf["name"] == "latent_softmax":
            from bert_ordinal.ordinal_models.experimental import (
                BertForWithLatentAndSoftMax,
            )

            model_cls = BertForWithLatentAndSoftMax

            def proc_logits(logits):
                label_dists = logits[0].softmax(dim=-1)
                return (
                    label_dists,
                    *summarize_label_dists(label_dists).values(),
                )

        elif model_conf["name"] == "threshold":
            from bert_ordinal.ordinal_models.experimental import (
                BertForMultiScaleThresholdRegression,
            )

            model_cls = BertForMultiScaleThresholdRegression
            proc_logits = cls.proc_logits_passthrough
        elif model_conf["name"] == "fixed_threshold":
            from bert_ordinal.ordinal_models.experimental import (
                BertForMultiScaleFixedThresholdRegression,
            )

            model_cls = BertForMultiScaleFixedThresholdRegression
            proc_logits = cls.proc_logits_passthrough
        elif model_conf["name"] == "metric":
            from bert_ordinal.ordinal_models.experimental import (
                BertForLatentScaleMetricLearning,
            )

            model_cls = BertForLatentScaleMetricLearning
            proc_logits = cls.proc_logits_passthrough
        elif model_conf["is_ordinal"]:
            from bert_ordinal import BertForMultiScaleOrdinalRegression

            model_cls = BertForMultiScaleOrdinalRegression
            model_kwargs = {
                **model_kwargs,
                "link": model_conf["link"],
                "discrimination_mode": args.discrimination_mode,
            }
        else:
            print(f"Unknown model type {args.model}", file=sys.stderr)
            sys.exit(-1)
        with silence_warnings():
            model = model_cls.from_pretrained(**model_kwargs)
        if model_conf["is_ordinal"]:
            link = model.link
            pred_proc = OrdinalPredProc(link)
        return model, label_names, pred_proc, proc_logits

    @property
    def dump_callback(self):
        return getattr(self.pred_proc, "dump_callback", None)

    @contextmanager
    def eval_dump_ctx(self, step):
        if self.args.dump_results:
            self.dump_writer.start_step_dump(step)
        try:
            yield
        finally:
            if self.args.dump_results:
                self.dump_writer.finish_step_dump()

    def decode_labels(self, labels):
        if len(self.label_names) == 2:
            labels, task_ids = labels
            batch_num_labels = np.empty(len(task_ids), dtype=np.int32)
            for idx, task_id in enumerate(task_ids):
                batch_num_labels[idx] = self.num_labels[task_id]
        else:
            task_ids = torch.zeros(len(labels), dtype=torch.uint8)
            batch_num_labels = torch.full(
                len(labels), self.num_labels, dtype=torch.uint8
            )
        return labels, task_ids, batch_num_labels

    def compute_metrics(self, eval_pred):
        pred_label_dists, labels = eval_pred
        labels, task_ids, batch_num_labels = self.decode_labels(labels)

        step = self.trainer.state.global_step
        if self.args.trace_labels_predictions:
            print()
            print(f"Step {step}")
            print("Computing metrics based upon")
            print("labels")
            print(labels)
            print("predictions")
            pprint(pred_label_dists)

        with self.eval_dump_ctx(step):
            res = {}
            preds = self.pred_proc.postprocess(
                pred_label_dists, batch_num_labels=batch_num_labels
            )
            if self.args.dump_results:
                self.dump_writer.add_info_full("test", **flatten_dump(preds))
            if "hidden" in preds:
                res.update(
                    self.refit_latent(
                        task_ids, preds["hidden"], batch_num_labels, labels
                    )
                )
            elif self.args.dump_results:
                # Refit latent is not called, so we need to dump the predictions on the train set here
                self.log_eval_on_train_data_class()
            if "pred" in preds:
                res.update(
                    evaluate_predictions(
                        preds["pred"], labels, batch_num_labels, task_ids
                    )
                )
            if "agg_preds" in preds:
                evaluate_pred_dist_avgs(
                    preds["agg_preds"], labels, batch_num_labels, task_ids
                )
                if self.model_conf["name"] != "class":
                    raise ValueError(
                        f"Do not know how to process metrics for {self.model_conf['name']}"
                    )
            self.dump_writer.add_heads("model", self.model)
            add_bests(res)
            return res

    def log_eval_on_train_data_class(self):
        for batch, result in inference_run(
            self.model,
            self.tokenizer,
            self.eval_dataset["train"],
            self.training_args.train_batch_size,
            eval_mode=True,
            use_tqdm=True,
            pass_task_ids=True,
            return_dict=False,
        ):
            result = torch.vstack(result)
            self.dump_writer.add_info_chunk(
                "train",
                **flatten_dump(
                    self.pred_proc.postprocess(
                        nested_numpify(self.pred_proc.proc_logits(result)),
                        batch.scale_points,
                    )
                ),
            )

    def refit_latent(self, task_ids, test_hiddens, batch_num_labels, test_labels):
        from bert_ordinal.eval import refit_eval

        print()
        print(" * Refit * ")
        print()
        print()
        return refit_eval(
            self.model,
            self.tokenizer,
            self.eval_dataset["train"],
            self.training_args.train_batch_size,
            task_ids,
            self.scale_points_map,
            self.train_hidden_buffer,
            test_hiddens,
            batch_num_labels,
            test_labels,
            self.regressor_buffers,
            dump_writer=self.dump_writer if self.args.dump_results else None,
            dump_callback=self.dump_callback,
            num_workers=self.args.num_vgam_workers,
            pool=self.pool,
            vglm_kwargs=dict(mask_vglm_errors=True, suppress_vglm_output=True),
        )

    def mk_eval_buffers(self):
        from collections import Counter

        regressor_buffers = {}
        cnts = Counter()
        for task_id in self.eval_dataset["train"]["task_ids"]:
            cnts[task_id] += 1
        regressor_buffers = {
            task_id: (np.empty(cnt), np.empty(cnt, dtype=np.uint8))
            for task_id, cnt in cnts.items()
        }

        scale_points_map = {}
        for ds in self.eval_dataset.values():
            for row in ds:
                scale_points_map[row["task_ids"]] = row["scale_points"]
        return (
            regressor_buffers,
            np.empty(len(self.eval_dataset["train"])),
            scale_points_map,
        )

    def get_preprocess_logits_for_metrics(self):
        proc_logits = None
        if self.pred_proc is not None:
            proc_logits = self.pred_proc.proc_logits
        if proc_logits is None and self.proc_logits is not None:
            proc_logits = self.proc_logits
        if proc_logits is None:
            return None
        return lambda logits, _labels: proc_logits(logits)

    def get_trainer(self):
        if self.model_conf["name"] == "metric":
            from bert_ordinal.ordinal_models.experimental import MetricLearningTrainer

            if self.args.sampler != "default":
                raise ValueError("Custom samplers not supported for metric learning")

            return MetricLearningTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.eval_test_dataset,
                compute_metrics=self.compute_metrics,
                preprocess_logits_for_metrics=self.get_preprocess_logits_for_metrics(),
                tokenizer=self.tokenizer,
                optimizers=self.optimizers,
            )
        else:
            if self.args.sampler == "default":
                self.training_args.group_by_length = True
                return Trainer(
                    model=self.model,
                    args=self.training_args,
                    train_dataset=self.dataset["train"],
                    eval_dataset=self.eval_dataset["test"],
                    compute_metrics=self.compute_metrics,
                    preprocess_logits_for_metrics=self.get_preprocess_logits_for_metrics(),
                    tokenizer=self.tokenizer,
                    optimizers=self.optimizers,
                )
            else:
                from bert_ordinal.ordinal_models.experimental import (
                    CustomSamplerTrainer,
                )

                return CustomSamplerTrainer(
                    sampler=self.args.sampler,
                    model=self.model,
                    args=self.training_args,
                    train_dataset=self.dataset["train"],
                    eval_dataset=self.eval_dataset["test"],
                    compute_metrics=self.compute_metrics,
                    preprocess_logits_for_metrics=self.get_preprocess_logits_for_metrics(),
                    tokenizer=self.tokenizer,
                    optimizers=self.optimizers,
                )

    def train(self):
        self.tokenizer = get_tokenizer()
        args = self.args
        training_args = self.training_args

        (
            self.regressor_buffers,
            self.train_hidden_buffer,
            self.scale_points_map,
        ) = self.mk_eval_buffers()

        init_weights(
            training_args,
            args,
            self.model_conf,
            self.model,
            self.tokenizer,
            self.eval_dataset,
        )

        print("")
        print(f" ** Training model {args.model} on dataset {args.dataset} ** ")
        print("")

        self.trainer = self.get_trainer()

        pool = None
        if args.dump_results:
            from joblib import Parallel

            dump_writer_cb = DumpWriterCallback(
                args.dump_results, zip_with=relpath(args.dataset, args.dump_results)
            )
            self.dump_writer = dump_writer_cb.dump_writer
            self.trainer.add_callback(dump_writer_cb)
            if args.num_vgam_workers == 0:
                num_workers = 1
            else:
                num_workers = args.num_vgam_workers
            pool = Parallel(num_workers, timeout=30)
        if args.dump_initial_model is not None:
            self.trainer.save_model(
                training_args.output_dir + "/" + args.dump_initial_model
            )
        with ensure_context(pool) as pool_ctx:
            self.pool = pool_ctx
            self.trainer.train()


def main():
    TrainerAndEvaluator().train()


if __name__ == "__main__":
    main()
