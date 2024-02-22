import threading

import numpy as np

local = threading.local()


def get_r_imports():
    from rpy2.robjects.packages import importr

    if not hasattr(local, "r_imports"):
        local.r_imports = (importr("VGAM"), importr("base"), importr("stats"))
    return local.r_imports


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

    vgam, base, stats = get_r_imports()

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

    if suppress_vglm_output:
        null_conn = base.textConnection(ro.NULL, "w")
        base.sink(null_conn, type="message")
        base.sink(null_conn)
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
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
    prev_src_idx = None
    for dest_idx, src_idx in enumerate(inv):
        if src_idx == -1:
            if dest_idx < represented_indices[0]:
                coefs_full[0, 0] = coefs[0, 0] - intercept_range
                coefs_full[1, 0] = coefs[1, 0]
            elif dest_idx > represented_indices[-1]:
                coefs_full[0, dest_idx] = coefs[0, -1] + intercept_range
                coefs_full[1, dest_idx] = coefs[1, -1]
            else:
                coefs_full[:, dest_idx] = coefs[:, inv[prev_src_idx]]
        else:
            coefs_full[:, dest_idx] = coefs[:, src_idx]
            prev_src_idx = src_idx
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


def prepare_regressors(
    train_hiddens_buffer,
    regressor_buffers,
    model,
    tokenizer,
    train_dataset,
    batch_size,
    dump_writer=None,
    dump_callback=None,
):
    from ms_text_regress.transformers_utils import inference_run

    pos = 0
    positions = {task_id: 0 for task_id in regressor_buffers}
    for batch, result in inference_run(
        model,
        tokenizer,
        train_dataset,
        batch_size,
        eval_mode=True,
        use_tqdm=True,
        pass_task_ids=dump_callback is not None,
    ):
        if dump_writer is not None:
            extra_kwargs = {}
            if dump_callback is not None:
                extra_kwargs = dump_callback(batch, result)
            dump_writer.add_info_chunk(
                "train",
                hidden=result.hidden_linear.detach().cpu().numpy().squeeze(-1),
                **extra_kwargs,
            )
        batch_info = zip(
            batch["task_ids"],
            batch["labels"],
            batch["scale_points"],
            result.hidden_linear,
        )
        for task_id, label, scale_points, hidden_linear in batch_info:
            task_id = task_id.item()
            label = label.item()
            hidden_linear = hidden_linear.item()
            xs, ys = regressor_buffers[task_id]
            train_hiddens_buffer[pos] = hidden_linear
            xs[positions[task_id]] = hidden_linear
            ys[positions[task_id]] = label
            positions[task_id] += 1
            pos += 1
    return train_hiddens_buffer, regressor_buffers


def fit_one_task(
    task_id, coefs, scale_points, family_name, mask_vglm_errors=False, **kwargs
):
    from pandas import DataFrame
    from rpy2.rinterface_lib.embedded import RRuntimeError

    xs, ys = coefs
    df = DataFrame({"xs": xs, "ys": ys})
    try:
        coefs = vglm(df, "xs", "ys", family_name, scale_points, **kwargs)
    except RRuntimeError:
        if not mask_vglm_errors:
            raise
        return task_id, None
    else:
        return task_id, coefs


def link_of_family_name(family_name):
    from ms_text_regress.element_link import get_link_by_name

    return get_link_by_name("fwd_" + family_name)


def label_dists_from_coefs(
    family_name,
    coefs,
    task_ids,
    hiddens,
    batch_num_labels,
):
    import torch

    link = link_of_family_name(family_name)
    label_dists = []
    for task_id, hidden, nl in zip(task_ids, hiddens, batch_num_labels, strict=True):
        if coefs.get(task_id) is None:
            label_dists.append(torch.ones((nl,)) / nl)
        else:
            intercepts = coefs[task_id][0, :]
            coef = coefs[task_id][1, :]
            logits = hidden.squeeze() * coef + intercepts
            label_dists.append(link.label_dist_from_logits(torch.tensor(logits)))
    return label_dists
