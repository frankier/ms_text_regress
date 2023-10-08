"""
This module implements evaluation metrics for ordinal regression tasks including
those with multiple scales.
"""
import sys
from functools import cache
from multiprocessing import TimeoutError

import numpy as np

try:
    import numba
except ModuleNotFoundError as err:
    raise RuntimeError("bert_ordinal.eval requires numba") from err


@numba.njit
def _jit_qwk(a1: np.ndarray, a2: np.ndarray, num_labels: int) -> float:
    hist1 = np.zeros((num_labels,))
    hist2 = np.zeros((num_labels,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(num_labels):
        for j in range(num_labels):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    if e == 0:
        return 0.0
    return 1 - o / (e / a1.shape[0])


def qwk(a1, a2, num_labels: int) -> float:
    """
    This method calculates the Quadratic Weighted Kappa between label arrays
    produced by two annotators -- or predicted and true labels -- `a1` and
    `a2`.

    Args:
        a (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 1. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        b (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 2. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        num_labels (`int`):
            The number of labels which bound the values in `a` and `b`.

    Attribution:
    This code is based on Apache-v2 licensed code (c) Jean-Francois Puget
    Source: https://www.kaggle.com/code/cpmpml/ultra-fast-qwk-calc-method/notebook
    """

    if len(a1) != len(a2):
        raise ValueError(
            f"Lengths of annotations passed to qwk must be equal {len(a1)} != {len(a2)}"
        )
    a1 = np.asarray(a1, dtype=np.int32)
    a2 = np.asarray(a2, dtype=np.int32)
    return _jit_qwk(a1, a2, num_labels)


@numba.njit
def calc_label_dist_lcm(num_labels):
    res = 1
    for nl in num_labels:
        res = np.lcm(nl - 1, res)
    return res


@numba.njit
def _jit_qwk_multi_norm(a1: np.ndarray, a2: np.ndarray, num_labels, label_dist_lcm):
    max_labels = np.max(num_labels)
    hist1 = np.zeros((max_labels,), dtype=np.int32)
    hist2 = np.zeros((max_labels,), dtype=np.int32)

    o_int = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        # Try and avoid floating point issues by keeping everything as integers
        # This makes hist1 and hist2 bigger by a constant factor label_dist_lcm each
        scaling_factor = label_dist_lcm // (num_labels[k] - 1)
        hist1[i] += scaling_factor
        hist2[j] += scaling_factor
        o_int += (i - j) ** 2 * scaling_factor**2

    e_int = 0
    for i in range(max_labels):
        for j in range(max_labels):
            # e is bigger by a factor of label_dist_lcm ** 2
            e_int += hist1[i] * hist2[j] * (i - j) * (i - j)

    # label_dist_lcm ** 2 factor and scaling_factor in general gets cancelled
    # out here
    return 1.0 - ((o_int * a1.shape[0]) / e_int)


def qwk_multi_norm(a1, a2, num_labels, label_dist_lcm=None):
    """
    This method calculates the Quadratic Weighted Kappa between label arrays
    produced by two annotators -- or predicted and true labels -- `a1` and
    `a2`. This one

    Args:
        a (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 1. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        b (`numpy.ndarray` of `int32` or convertible iterable)
            An integer array of labels from annotator 2. Each labels should be
            in the range half-open, Python-style range [0, num_labels).
        num_labels (`numpy.ndarray` of `int32` or convertible iterable)
            The number of labels
        label_dist_lcm (`numpy.ndarray`, *optional*):
            A common multiple of `num_labels[i] - 1` for all `num_labels[i]`.
            This is typically the Lowest Common Multiple (LCM). If not
            presupplied it will be calculated based on `num_labels`.

    Attribution:
    This code is based on Apache-v2 licensed code (c) Jean-Francois Puget
    Source: https://www.kaggle.com/code/cpmpml/ultra-fast-qwk-calc-method/notebook
    """

    if len(a1) != len(a2):
        raise ValueError(
            f"Lengths of annotations passed to qwk_multi_norm must be equal {len(a1)} != {len(a2)}"
        )
    a1 = np.asarray(a1, dtype=np.int32)
    a2 = np.asarray(a2, dtype=np.int32)
    num_labels = np.asarray(num_labels, dtype=np.int32)
    if label_dist_lcm is None:
        label_dist_lcm = calc_label_dist_lcm(num_labels)
    return _jit_qwk_multi_norm(a1, a2, num_labels, label_dist_lcm)


@cache
def evaluate_metrics():
    import evaluate

    metric_accuracy = evaluate.load("accuracy")
    metric_mae = evaluate.load("mae")
    metric_mse = evaluate.load("mse")
    return (metric_accuracy, metric_mae, metric_mse)


@numba.njit
def _jit_ms_mae(predictions, references, num_labels):
    # unique, indices = np.unique(num_labels, return_inverse=True)
    # true_positives = np.zeros(len(unique), type=np.int32)
    # totals = np.zeros(len(unique), type=np.int32)
    acc = 0.0
    for pred, ref, nl in zip(predictions, references, num_labels):
        acc += abs((pred - ref)) / (nl - 1)
    return acc / len(predictions)


def ms_mae(predictions, reference, num_labels):
    if len(predictions) != len(reference) or len(reference) != len(num_labels):
        raise ValueError(
            "Lengths of predictions, reference and num_labels passed to ms_mae must be equal"
        )
    predictions = np.asarray(predictions, dtype=np.int32)
    reference = np.asarray(reference, dtype=np.int32)
    num_labels = np.asarray(num_labels, dtype=np.int32)
    return _jit_ms_mae(predictions, reference, num_labels)


def basic_metrics(predictions, labels, num_labels):
    metric_accuracy, metric_mae, metric_mse = evaluate_metrics()
    mse = metric_mse.compute(predictions=predictions, references=labels)
    return {
        **metric_accuracy.compute(predictions=predictions, references=labels),
        **metric_mae.compute(predictions=predictions, references=labels),
        **mse,
        "rmse": (mse["mse"]) ** 0.5,
        "ms_mae": ms_mae(predictions, labels, num_labels),
    }


def evaluate_predictions(predictions, labels, num_labels, task_ids=None):
    metrics = basic_metrics(predictions, labels, num_labels)
    if task_ids is not None:
        sorted_idx = np.argsort(task_ids)
        cuts = np.unique(task_ids[sorted_idx], return_index=True)[1][1:]
        tavg_metrics = {}
        for pred, lbl, nl in zip(
            np.split(predictions[sorted_idx], cuts),
            np.split(labels[sorted_idx], cuts),
            np.split(num_labels[sorted_idx], cuts),
        ):
            for name, metric in basic_metrics(pred, lbl, nl).items():
                if name not in tavg_metrics:
                    tavg_metrics[name] = 0.0
                tavg_metrics[name] += metric
            if "qwk" not in tavg_metrics:
                tavg_metrics["qwk"] = 0.0
            tavg_metrics["qwk"] += qwk(pred, lbl, nl[0])
        for metric, val in tavg_metrics.items():
            metrics[f"tavg/{metric}"] = val / (len(cuts) + 1)
    return metrics


def evaluate_pred_dist_avgs(pred_dist_avgs, labels, num_labels, task_ids=None):
    from .label_dist import PRED_AVGS

    res = {}
    for avg in PRED_AVGS:
        for k, v in evaluate_predictions(
            pred_dist_avgs[avg], labels, num_labels, task_ids
        ).items():
            res[f"{avg}/{k}"] = v
    return res


def _tagdelay_helper(name, f, *args, **kwargs):
    return (name, f(*args, **kwargs))


def tagdelay(name, f, *args, **kwargs):
    from joblib import delayed

    return delayed(_tagdelay_helper)(name, f, *args, **kwargs)


def generate_refits(regressors, scale_points_map, vglm_kwargs):
    from bert_ordinal.baseline_models.skl_wrap import fit
    from bert_ordinal.ordinal_models.vglm import fit_one_task

    for task_id, coefs in regressors.items():
        yield tagdelay(
            "cumulative",
            fit_one_task,
            task_id,
            coefs,
            scale_points_map[task_id],
            "cumulative",
            **vglm_kwargs,
        )
    for task_id, coefs in regressors.items():
        yield tagdelay(
            "acat",
            fit_one_task,
            task_id,
            coefs,
            scale_points_map[task_id],
            "acat",
            **vglm_kwargs,
        )
    for task_id, coefs in regressors.items():
        yield tagdelay("linear", fit, task_id, coefs)


def ensure_pool(num_workers=0, pool=None):
    from joblib import Parallel

    if num_workers == 0:
        num_workers = 1
    if pool is None:
        return Parallel(num_workers, timeout=30)
    return pool


def pred_and_dump(
    family_name,
    split_name,
    coefs,
    hiddens,
    task_ids,
    batch_num_labels,
    dump_writer=None,
):
    if family_name == "linear":
        from bert_ordinal.baseline_models.skl_wrap import predict

        preds = predict(
            coefs,
            task_ids,
            hiddens,
            batch_num_labels,
        )

        if dump_writer is not None:
            dump_writer.add_info_full(
                split_name,
                **{"pred/refit/linear": preds},
            )
        return preds
    else:
        from bert_ordinal.label_dist import summarize_label_dists
        from bert_ordinal.ordinal_models.vglm import label_dists_from_coefs

        label_dists = label_dists_from_coefs(
            family_name,
            coefs,
            task_ids,
            hiddens,
            batch_num_labels,
        )
        summarized_label_dists = summarize_label_dists(label_dists)
        if dump_writer is not None:
            dump_writer.add_info_full(
                split_name,
                **{
                    f"label_dists/refit/{family_name}": [
                        ld.cpu().numpy() for ld in label_dists
                    ]
                },
            )
            dump_writer.add_info_full(
                split_name,
                **{
                    f"pred/refit/{family_name}/{avg}": v.cpu().numpy()
                    for avg, v in summarized_label_dists.items()
                },
            )
        return summarized_label_dists


def dump_refit_heads(family_name, model, coefs, dump_writer=None):
    if dump_writer is None:
        return
    dump_writer.add_heads("refit/" + family_name, model, coefs)


def refit_eval(
    model,
    tokenizer,
    train_dataset,
    batch_size,
    task_ids,
    scale_points_map,
    train_hiddens_buffer,
    test_hiddens,
    batch_num_labels,
    test_labels,
    regressor_buffers,
    dump_writer=None,
    dump_callback=None,
    num_workers=0,
    pool=None,
    vglm_kwargs=None,
):
    if vglm_kwargs is None:
        vglm_kwargs = {}
    from tqdm.auto import tqdm

    from bert_ordinal.ordinal_models.vglm import prepare_regressors

    res = {}
    train_hiddens, regressors = prepare_regressors(
        train_hiddens_buffer,
        regressor_buffers,
        model,
        tokenizer,
        train_dataset,
        batch_size,
        dump_writer=dump_writer,
        dump_callback=dump_callback,
    )
    pool = ensure_pool(num_workers, pool)
    all_coefs = {}
    total = len(regressors) * 3
    count = 0
    bar = tqdm(total=total)
    try:
        for typ, payload in pool(
            generate_refits(regressors, scale_points_map, vglm_kwargs)
        ):
            bar.update(1)
            count += 1
            task_id, coefs = payload
            all_coefs.setdefault(typ, {})[task_id] = coefs
    except TimeoutError:
        print(
            f"WARNING: Timed out during refits with {count}/{total} tasks successful",
            file=sys.stderr,
        )

    to_pred = [("test", test_hiddens, task_ids, batch_num_labels)]
    if dump_writer is not None:
        to_pred.append(
            (
                "train",
                train_hiddens,
                train_dataset["task_ids"],
                train_dataset["scale_points"],
            )
        )

    for family_name, coefs in all_coefs.items():
        dump_refit_heads(family_name, model, coefs, dump_writer=dump_writer)
        for split_name, hiddens, tids, nls in to_pred:
            preds = pred_and_dump(
                family_name,
                split_name,
                coefs,
                hiddens,
                tids,
                nls,
                dump_writer=dump_writer,
            )
            if split_name == "test":
                if family_name == "linear":
                    eval = evaluate_predictions(preds, test_labels, nls, tids)
                else:
                    eval = evaluate_pred_dist_avgs(preds, test_labels, nls, tids)
                for k, v in eval.items():
                    res[f"refit/{family_name}/{k}"] = v
    return res


def add_bests(metrics):
    best_ms_mae = float("inf")
    best_acc = float("inf")
    best_tavg_qwk = float("inf")
    for k, v in metrics.items():
        bits = k.split("/")
        if bits[-1] == "ms_mae":
            best_ms_mae = min(best_ms_mae, v)
        elif bits[-1] == "acc":
            best_acc = max(best_acc, v)
        elif len(bits) >= 2 and bits[-2:] == ["tavg", "qwk"]:
            best_tavg_qwk = max(best_tavg_qwk, v)
    metrics["best/ms_mae"] = best_ms_mae
    metrics["best/acc"] = best_acc
    metrics["best/tavg/qwk"] = best_tavg_qwk
    return metrics
