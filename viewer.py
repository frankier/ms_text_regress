import argparse
import json
import pickle

import altair as alt
import numpy
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
def load_data(path):
    with open(path) as f:
        records = [json.loads(line) for line in f]
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
                    *PRED_AVGS,
                    *(["hidden"] if "hidden" in records[0] else []),
                ]
            }
            for row in records
        )
        return records, df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Input file of an eval dump")
    parser.add_argument("--thresholds", help="Input task thresholds pickle")
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


def plot_conf_mat(outputs, targets):
    confmat_data = confusion_matrix(targets, outputs)
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


def main():
    args = parse_args()

    if args.path:
        records, df = load_data(args.path)
        selection = aggrid_interactive_table(df)

        def get_selected_records():
            for row in selection.selected_rows:
                yield int(row["_selectedRowNodeInfo"]["nodeId"])

        selected_rows = selection.selected_rows
    else:
        selected_rows = None

    if args.thresholds is not None:
        task_infos = get_task_infos(args.thresholds)
    else:
        task_infos = None

    if selected_rows and len(selected_rows) == 1:
        selected_record = records[next(get_selected_records())]
        score_chart = plot_score_dist(selected_record["scores"])
        if "el_mo_summary" in selected_record:
            el_mo_chart = plot_el_mo_dist(selected_record["el_mo_summary"])
        else:
            el_mo_chart = None
        if task_infos is not None:
            task_info = task_infos[selected_record["task_ids"]]
        else:
            task_info = None
        st.json(
            {
                k: v
                for k, v in selected_record.items()
                if k not in ["scores", "el_mo_summary"]
            }
        )
        col1, col2 = st.columns(2)
        col1.altair_chart(score_chart.interactive(), use_container_width=True)
        if el_mo_chart is not None:
            col2.altair_chart(el_mo_chart.interactive(), use_container_width=True)
        if task_info is not None:
            plot_task(task_info, selected_record["hidden"])
    elif selected_rows and len(selected_rows) > 1:
        selected_df = df.iloc[get_selected_records()]
        for avg in ["median", "mode"]:
            st.altair_chart(
                plot_conf_mat(selected_df[avg], selected_df["label"]).properties(
                    title=avg.title()
                )
            )
        # XXX: mean, hidden, sum all label dists
    else:
        st.text(
            "Select one row to analyse a single prediction, and more than one rows to compare predictions. "
            "You can inspect task thresholds below."
        )
        if task_infos is not None:
            selected = st.selectbox("Task", range(len(task_infos)))

            plot_task(task_infos[selected])


if __name__ == "__main__":
    main()
