import argparse
import json

import altair as alt
import pandas
import streamlit as st
import torch

from bert_ordinal import BertForMultiScaleOrdinalRegression

st.set_page_config(
    page_title="Ordinal classification results browser",
    page_icon="📊",
)

LOGIT_99 = torch.logit(torch.tensor(0.99))


@st.experimental_singleton()
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
                ]
            }
            for row in records
        )
        return records, df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Input file of an eval dump", required=True)
    parser.add_argument("--model", help="Input directory of model dump", required=True)
    return parser.parse_args()


def aggrid_interactive_table(df: pandas.DataFrame):
    from st_aggrid import AgGrid, GridOptionsBuilder
    from st_aggrid.shared import ColumnsAutoSizeMode, GridUpdateMode

    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    options.configure_selection("single")
    with st.sidebar:
        selection = AgGrid(
            df,
            gridOptions=options.build(),
            theme="streamlit",
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
        )

    return selection


@st.experimental_singleton()
def get_task_infos(model_path):
    with torch.inference_mode():
        print("Loading model")
        model = BertForMultiScaleOrdinalRegression.from_pretrained(model_path)
        print("Loaded")
        task_outs = []
        for task_id in range(len(model.num_labels)):
            discrimination, offsets = model.cutoffs.task_summary(task_id)
            min_latent = (offsets - LOGIT_99 / discrimination).min()
            max_latent = (offsets + LOGIT_99 / discrimination).max()
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
            task_outs.append(task_info_long)
    return task_outs


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


def plot_hidden_dist(task_info, hidden):
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
    hidden_mark = (
        alt.Chart(alt.Data(values=[{"hidden": hidden}]))
        .mark_rule(color="black")
        .encode(x="hidden:Q")
    )
    return lines + hidden_mark


def main():
    args = parse_args()
    records, df = load_data(args.path)
    task_infos = get_task_infos(args.model)
    selection = aggrid_interactive_table(df)
    if selection.selected_rows and len(selection.selected_rows) == 1:
        selected_id = selection.selected_rows[0]["_selectedRowNodeInfo"]["nodeId"]
        selected_record = records[int(selected_id)]
        score_chart = plot_score_dist(selected_record["scores"])
        el_mo_chart = plot_el_mo_dist(selected_record["el_mo_summary"])
        task_info = task_infos[selected_record["task_ids"]]
        st.json(
            {
                k: v
                for k, v in selected_record.items()
                if k not in ["scores", "el_mo_summary"]
            }
        )
        col1, col2 = st.columns(2)
        col1.altair_chart(score_chart.interactive(), use_container_width=True)
        col2.altair_chart(el_mo_chart.interactive(), use_container_width=True)
        st.altair_chart(
            plot_hidden_dist(task_info, selected_record["hidden"]).interactive(),
            use_container_width=True,
        )
    else:
        st.header(
            "Select 1 row to analyse a single prediction, and >1 rows to compare predictions"
        )


if __name__ == "__main__":
    main()
