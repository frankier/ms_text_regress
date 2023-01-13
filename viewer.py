import argparse
import pickle
from os.path import join as pjoin

import altair as alt
import numpy
import orjson
import pandas
import streamlit as st
import torch
from sklearn.metrics import confusion_matrix

from bert_ordinal.label_dist import PRED_AVGS

st.set_page_config(
    page_title="Ordinal classification results browser",
    page_icon="📊",
)

LOGIT_99 = torch.logit(torch.tensor(0.99))


@st.experimental_memo()
def load_data(path, zip_with=None, zip_with_seg=None):
    if zip_with is not None:
        import datasets

        zip_with_data = datasets.load_from_disk(zip_with)[zip_with_seg]
        with open(path, "rb") as f:
            records = [
                {**orjson.loads(line), **rec} for line, rec in zip(f, zip_with_data)
            ]
    else:
        with open(path, "rb") as f:
            records = [orjson.loads(line) for line in f]
    df = pandas.DataFrame(
        {
            k: row[k]
            for k in [
                "movie_title",
                "text",
                "label",
                "scale_points",
                "review_score",
                "task_ids",
                *(k for k in [*PRED_AVGS, "hidden"] if k in records[0]),
            ]
        }
        for row in records
    )
    return records, df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Input file of an eval dump")
    parser.add_argument("--thresholds", help="Input task thresholds pickle")
    parser.add_argument("--multi", help="Multi-checkpoint dump")
    parser.add_argument("--zip-with")
    parser.add_argument("--zip-with-seg")

    return parser.parse_args()


def aggrid_interactive_table(df: pandas.DataFrame):
    from st_aggrid import AgGrid, GridOptionsBuilder
    from st_aggrid.shared import ColumnsAutoSizeMode, GridUpdateMode

    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    options.configure_selection("multiple")
    options.configure_pagination(enabled=True, paginationPageSize=100)
    options.configure_column(
        "movie_title",
        headerCheckboxSelection=True,
        headerCheckboxSelectionFilteredOnly=True,
        checkboxSelection=True,
        groupSelectsChildren=True,
        groupSelectsFiltered=True,
    )
    with st.sidebar:
        selection = AgGrid(
            df,
            gridOptions=options.build(),
            theme="streamlit",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        )

    return selection


@st.experimental_memo()
def get_task_infos(thresholds):
    with open(thresholds, "rb") as f:
        return pickle.load(f)


def plot_score_dist(scores):
    return alt.Chart(pandas.DataFrame(scores)).mark_bar().encode(x="index", y="score")


def plot_el_mo_dist(el_mo_summary):
    return (
        alt.Chart(pandas.DataFrame(el_mo_summary))
        .mark_bar()
        .encode(
            order=alt.Order("index", sort="ascending"),
            x=alt.X(
                "subprob",
                type="nominal",
                sort=alt.EncodingSortField("index", order="ascending"),
            ),
            y="score",
        )
    )


def plot_hidden_dist(task_info, hidden=None):
    lines = (
        alt.Chart(task_info)
        .mark_line()
        .encode(
            x="x",
            y="score",
            tooltip="subprob",
            color=alt.Color(
                "subprob", sort=alt.EncodingSortField("index", order="ascending")
            ),
        )
    )
    if hidden is not None:
        hidden_mark = (
            alt.Chart(alt.Data(values=[{"hidden": hidden}]))
            .mark_rule(color="black")
            .encode(x="hidden:Q")
        )
        return lines + hidden_mark
    else:
        return lines


def plot_hidden_mono_func(task_info, hidden=None):
    lines = (
        alt.Chart(task_info)
        .mark_line()
        .encode(
            x="x",
            y="score",
        )
    )
    if hidden is not None:
        hidden_mark = (
            alt.Chart(alt.Data(values=[{"hidden": hidden}]))
            .mark_rule(color="black")
            .encode(x="hidden:Q")
        )
        return lines + hidden_mark
    else:
        return lines


def melt_conf_mat(confmat):
    trues = []
    preds = []
    cnts = []
    for idx, val in numpy.ndenumerate(confmat):
        trues.append(idx[0])
        preds.append(idx[1])
        cnts.append(val)
    return pandas.DataFrame(
        {
            "true": trues,
            "pred": preds,
            "cnt": cnts,
        }
    )


def plot_conf_mat(outputs, targets, scale_points):
    confmat_data = confusion_matrix(targets, outputs, labels=range(scale_points))
    true_scale = alt.Scale(domain=list(range(len(confmat_data))))
    pred_scale = alt.Scale(domain=list(range(len(confmat_data))))
    confmat_long = melt_conf_mat(confmat_data)
    base_chart = alt.Chart(confmat_long)
    confmat_chart = base_chart.mark_rect().encode(
        alt.X("true:O"), alt.Y("pred:O"), color="cnt:Q"
    )
    top_hist = (
        base_chart.mark_bar()
        .encode(
            alt.X("true:O", title="", scale=true_scale),
            alt.Y("sum(cnt)", title=""),
        )
        .properties(height=60)
    )
    right_hist = (
        base_chart.mark_bar()
        .encode(
            alt.Y("pred:O", title="", scale=pred_scale),
            alt.X("sum(cnt)", title=""),
        )
        .properties(width=60)
    )

    return top_hist & (confmat_chart | right_hist)


def plot_task(task_info, hidden=None):
    if "hidden_to_elmo" in task_info:
        st.altair_chart(
            plot_hidden_dist(task_info["hidden_to_elmo"], hidden).interactive(),
            use_container_width=True,
        )
        st.json(
            {
                "discrimination": task_info["discrimination"],
                "offsets": task_info["offsets"],
            }
        )
    else:
        st.altair_chart(
            plot_hidden_mono_func(task_info["hidden_to_label"], hidden).interactive(),
            use_container_width=True,
        )


AVGS = ["median", "mode", "mean"]


def multi_conf_mat(selected_df):
    from bert_ordinal.eval import evaluate_predictions

    known_ags = [avg for avg in AVGS if avg in selected_df.columns]
    if not known_ags:
        return
    tabs = st.tabs(known_ags)
    for avg, tab in zip(known_ags, tabs):
        with tab:
            st.altair_chart(
                plot_conf_mat(
                    selected_df[avg],
                    selected_df["label"],
                    selected_df["scale_points"].max(),
                ).properties(title=avg.title())
            )
            if st.button("Calculate metrics", key=avg):
                predictions = selected_df[avg].to_numpy()
                targets = selected_df["label"].to_numpy()
                scale_points = selected_df["scale_points"].to_numpy()
                st.json(evaluate_predictions(predictions, targets, scale_points))


def label_densities(df):
    show_counts = st.checkbox("Counts", value=True)
    densities = alt.Chart(df).transform_density(
        "hidden", groupby=["label"], counts=show_counts, as_=["hidden", "density"]
    )
    areas = densities.mark_area(line={"color": "black"}, opacity=0.5).encode(
        x="hidden:Q",
        y="density:Q",
        order="label:N",
    )
    labels = (
        densities.mark_text(fontSize=20, dy=-10)
        .transform_aggregate(
            max_density="max(density)",
            max_density_pt="argmax(density)",
            groupby=["label"],
        )
        .transform_calculate(
            hidden_at_max_density="datum.max_density_pt.hidden",
        )
        .encode(x="hidden_at_max_density:Q", y="max_density:Q", text="label:O")
    )

    st.altair_chart(areas + labels)


def get_review_score_map(df):
    mapping = {}
    for idx, grp in df.groupby(["label", "review_score"]):
        mapping[int(grp.iloc[0]["label"])] = grp.iloc[0]["review_score"]
    return mapping


def plot_paths(dump_path, thresholds_path, zip_with=None, zip_with_seg=None):
    selected_rows = None
    if dump_path:
        records, df = load_data(dump_path, zip_with, zip_with_seg)
        selection = aggrid_interactive_table(df)

        def get_selected_records():
            for row in selection.selected_rows:
                yield int(row["_selectedRowNodeInfo"]["nodeId"])

        selected_rows = selection.selected_rows

    task_infos = None
    if thresholds_path is not None:
        task_infos = get_task_infos(thresholds_path)

    if selected_rows and len(selected_rows) == 1:
        selected_record = records[next(get_selected_records())]
        score_chart = None
        if "scores" in selected_record:
            score_chart = plot_score_dist(selected_record["scores"])
        el_mo_chart = None
        if "el_mo_summary" in selected_record:
            el_mo_chart = plot_el_mo_dist(selected_record["el_mo_summary"])
        task_info = None
        if task_infos is not None:
            task_info = task_infos[selected_record["task_ids"]]
        st.json(
            {
                k: v
                for k, v in selected_record.items()
                if k not in ["scores", "el_mo_summary"]
            }
        )
        col1, col2 = st.columns(2)
        if score_chart is not None:
            col1.altair_chart(score_chart.interactive(), use_container_width=True)
        if el_mo_chart is not None:
            col2.altair_chart(el_mo_chart.interactive(), use_container_width=True)
        if task_info is not None:
            plot_task(task_info, selected_record["hidden"])
    elif selected_rows and len(selected_rows) > 1:
        selected_df = df.iloc[get_selected_records()]
        group_by_task = st.checkbox("Group by task")
        if group_by_task:
            task_id = st.selectbox("Task", selected_df["task_ids"].unique())
            selected_task_df = selected_df[selected_df["task_ids"] == task_id]
            st.json(get_review_score_map(selected_task_df), expanded=False)
            with st.expander("Confusion matrices", expanded=True):
                multi_conf_mat(selected_task_df)
            with st.expander("Label densities", expanded=True):
                label_densities(selected_task_df)
        else:
            multi_conf_mat(selected_df)
            label_densities(selected_df)
        # TODO: swarm plots
    else:
        st.text(
            "Select one row to analyse a single prediction, and more than one rows to compare predictions. "
            "You can inspect task thresholds below."
        )
        if task_infos is not None:
            selected = st.selectbox("Task", range(len(task_infos)))

            plot_task(task_infos[selected])


@st.experimental_memo()
def load_checkpoint_index(path):
    with open(pjoin(path, "index.json"), "rb") as f:
        index = orjson.loads(f.read())
    chkpt_dicts = {
        ds_split: {chkpt["nick"]: chkpt for chkpt in index[ds_split]}
        for ds_split in index
        if not ds_split.startswith("_")
    }
    return chkpt_dicts, index.get("_zip_with")


def main():
    args = parse_args()
    if args.multi:
        chkpt_dicts, zip_with = load_checkpoint_index(args.multi)
        if zip_with is None:
            zip_with = args.zip_with
        with st.sidebar:
            ds_split = st.selectbox("Dataset split", list(chkpt_dicts.keys()))
            chkpt_dict = chkpt_dicts[ds_split]
            checkpoint = st.select_slider("Checkpoint", list(chkpt_dict.keys()))
        selected_checkpoint = chkpt_dict[checkpoint]
        plot_paths(
            pjoin(args.multi, selected_checkpoint["dump"])
            if "dump" in selected_checkpoint
            else None,
            pjoin(args.multi, selected_checkpoint["thresholds"])
            if "thresholds" in selected_checkpoint
            else None,
            zip_with=pjoin(args.multi, zip_with) if zip_with is not None else None,
            zip_with_seg=ds_split,
        )
    else:
        plot_paths(
            args.path,
            args.thresholds,
            zip_with=args.zip_with,
            zip_with_seg=args.zip_with_seg,
        )


if __name__ == "__main__":
    main()
