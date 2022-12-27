import numpy as np


def vglm(
    df,
    x_var,
    y_var,
    family_name,
    num_labels=None,
    suppress_vglm_output=False,
    do_fill_missing_coefs=True,
    do_plus_one_smoothing=False,
):
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    vgam = importr("VGAM")
    base = importr("base")
    stats = importr("stats")

    if family_name == "cumulative_parallel":
        family = vgam.cumulative(parallel=True)
    elif family_name == "cumulative":
        family = vgam.cumulative(parallel=False)
    elif family_name == "acat_parallel":
        family = vgam.acat(parallel=True)
    elif family_name == "acat":
        family = vgam.acat(parallel=False)
    else:
        raise ValueError(f"Unknown family {family_name}")

    if do_plus_one_smoothing:
        df = do_plus_one_smoothing(df, y_var, num_labels)

    with localconverter(ro.default_converter + pandas2ri.converter):
        if suppress_vglm_output:
            base.sink("/dev/null", type="message")
            base.sink("/dev/null")
        try:
            result = vgam.vglm(f"{y_var} ~ {x_var}", data=df, family=family)
        finally:
            if suppress_vglm_output:
                base.sink(type="message")
                base.sink()
    coefs_r = stats.coef(result, matrix=True)
    coefs = np.asarray(coefs_r)
    if do_fill_missing_coefs:
        return fill_missing_coefs(coefs, df[y_var], num_labels)


def fill_missing_coefs(coefs, labels, num_labels=None):
    represented_indices = labels.sort_values().unique()
    if num_labels is None:
        num_labels = represented_indices[-1] + 1
    inv = np.full(num_labels - 1, -1, dtype=np.int)
    for i, x in enumerate(represented_indices[:-1]):
        inv[x] = i
    coefs_full = np.empty((2, num_labels - 1), dtype=np.float)
    intercept_range = coefs[0, -1] - coefs[0, 0]
    for dest_idx, src_idx in enumerate(inv):
        if src_idx == -1:
            if dest_idx == 0:
                coefs_full[0, 0] = coefs[0, 0] - intercept_range
                coefs_full[1, 0] = coefs[1, 0]
            else:
                cand_src_idx_idx = dest_idx
                while 1:
                    cand_src_idx_idx += 1
                    if cand_src_idx_idx >= num_labels - 1:
                        coefs_full[0, dest_idx] = coefs[0, -1] + intercept_range
                        coefs_full[1, dest_idx] = coefs[1, -1]
                        break
                    elif inv[cand_src_idx_idx] != -1:
                        coefs_full[:, dest_idx] = coefs[:, inv[cand_src_idx_idx]]
                        break
        else:
            coefs_full[:, dest_idx] = coefs[:, src_idx]
    return coefs_full


def plus_one_smoothing(df, x_var, y_var, num_labels):
    import pandas as pd

    xs = []
    ys = []
    for level in range(num_labels):
        if (df[y_var] == level).any():
            continue
        prev_idx = None
        for i in range(level - 1, -1, -1):
            if (df[y_var] == i).any():
                prev_idx = i
                break
        next_idx = None
        for i in range(level + 1, num_labels):
            if (df[y_var] == i).any():
                next_idx = i
                break
        if prev_idx is None:
            fake_hidden = df[df[y_var] == next_idx, "hidden"].min()
        elif next_idx is None:
            fake_hidden = df[df[y_var] == prev_idx, "hidden"].max()
        else:
            fake_hidden = (
                df[df[y_var] == next_idx, "hidden"].mean()
                + df[df[y_var] == prev_idx, "hidden"].mean()
            ) / 2
        xs.append(level)
        ys.append(fake_hidden)
    return pd.concat([df, pd.DataFrame({x_var: xs, y_var: ys})])


def refit(
    model, family_name, train_dataset, batch_size, mask_vglm_errors=False, **kwargs
):
    from pandas import DataFrame
    from rpy2.rinterface_lib.embedded import RRuntimeError

    from bert_ordinal.transformers_utils import inference_run

    results = {}
    for result_batch, idx_slice in inference_run(
        model, train_dataset, batch_size, eval_mode=True, yield_indices=True
    ):
        batch_info = zip(
            train_dataset["task_ids"][idx_slice],
            train_dataset["label"][idx_slice],
            train_dataset["scale_points"][idx_slice],
            result_batch.hidden_linear,
        )
        for task_id, label, scale_points, hidden_linear in batch_info:
            xs, ys, _ = results.setdefault(task_id, ([], [], scale_points))
            xs.append(hidden_linear.item())
            ys.append(label)
    for task_id, (xs, ys, scale_points) in results.items():
        df = DataFrame({"xs": xs, "ys": ys})
        try:
            coefs = vglm(df, "xs", "ys", family_name, scale_points, **kwargs)
        except RRuntimeError:
            if not mask_vglm_errors:
                raise
            results[task_id] = None
        else:
            results[task_id] = coefs
    return results


def label_dists_from_hiddens(
    model,
    family_name,
    train_dataset,
    batch_size,
    task_ids,
    test_hiddens,
    num_labels,
    **kwargs,
):
    import torch

    from bert_ordinal.element_link import get_link_by_name

    link = get_link_by_name("fwd_" + family_name)
    coefs = refit(model, family_name, train_dataset, batch_size, mask_vglm_errors=True)
    label_dists = []
    for task_id, test_hidden in zip(task_ids, test_hiddens, strict=True):
        if coefs[task_id] is None:
            nl = num_labels[task_id]
            label_dists.append(torch.ones((nl,)) / nl)
        else:
            intercepts = coefs[task_id][0, :]
            coef = coefs[task_id][1, :]
            logits = test_hidden.squeeze() * coef + intercepts
            label_dists.append(link.label_dist_from_logits(torch.tensor(logits)))
    return label_dists
