import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # so imports work

import gradio as gr
import plotly.graph_objects as go
from eval.metrics import load_results, summary_table
from data.streams import drift_blocks_for
from common.contract import STREAM_KINDS, METHOD_NAMES

SOURCES = {"MOCK (fake data)": "results/mock", "REAL results": "results"}


def _load(source):
    try:
        return load_results(SOURCES[source])
    except FileNotFoundError:
        raise gr.Error(f"No result files found in {SOURCES[source]}")


def replay_plot(source, stream, seed, methods):
    res = _load(source)
    d = res[res.stream == stream]
    if seed != "mean of all seeds":
        d = d[d.seed == int(seed)]
    if d.empty:
        have = sorted(set(zip(res.stream, res.seed)))
        raise gr.Error(
            f"No results for stream={stream}, seed={seed}. Available: {have}"
        )
    missing = [m for m in methods if m not in set(d.method)]
    if len(missing) == len(methods):
        raise gr.Error(
            f"None of the selected methods have results here. "
            f"Available: {sorted(set(d.method))}"
        )
    if missing:
        gr.Warning(f"No results yet for: {', '.join(missing)}")

    fig = go.Figure()
    for m in methods:
        g = d[d.method == m]
        if g.empty:
            continue
        agg = g.groupby("block")["rmse"].agg(["mean", "std"]).reset_index().fillna(0)
        fig.add_trace(go.Scatter(x=agg.block, y=agg["mean"], mode="lines", name=m))
        if seed == "mean of all seeds" and g.seed.nunique() > 1:
            fig.add_trace(
                go.Scatter(
                    x=list(agg.block) + list(agg.block[::-1]),
                    y=list(agg["mean"] + agg["std"])
                    + list((agg["mean"] - agg["std"])[::-1]),
                    fill="toself",
                    line=dict(width=0),
                    opacity=0.15,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fired = g[g.drift_flag == 1].groupby("block").size().reset_index()
        if len(fired):
            fv = agg[agg.block.isin(fired.block)]
            fig.add_trace(
                go.Scatter(
                    x=fv.block,
                    y=fv["mean"],
                    mode="markers",
                    marker=dict(symbol="x", size=12),
                    name=f"{m}: detector fired",
                )
            )
    for b in drift_blocks_for(stream):
        fig.add_vline(x=b, line_dash="dash", annotation_text="drift starts")
    tag = "[MOCK DATA] " if source.startswith("MOCK") else ""
    fig.update_layout(
        title=f"{tag}Block RMSE ({stream}, {seed})",
        xaxis_title="Block",
        yaxis_title="RMSE",
        showlegend=True,
    )
    return fig


def summary(source, stream):
    tab = summary_table(_load(source))
    return tab[tab.stream == stream].drop(columns="stream").round(2)


with gr.Blocks(title="TabPFN under concept drift") as demo:
    gr.Markdown("# TabPFN under concept drift: context-selection policies")
    source = gr.Radio(list(SOURCES), value="MOCK (fake data)", label="Data source")

    with gr.Tab("Replay"):
        with gr.Row():
            stream = gr.Dropdown(list(STREAM_KINDS), value="sudden", label="Stream")
            seed = gr.Dropdown(
                [0, 1, 2, "mean of all seeds"], value="mean of all seeds", label="Seed"
            )
        methods = gr.CheckboxGroup(
            METHOD_NAMES, value=["P0_vanilla", "P1_window", "XGBoost"], label="Methods"
        )
        plot = gr.Plot()
        gr.Button("Plot").click(replay_plot, [source, stream, seed, methods], plot)

    with gr.Tab("Summary table"):
        s_stream = gr.Dropdown(list(STREAM_KINDS), value="sudden", label="Stream")
        table = gr.Dataframe()
        gr.Button("Show").click(summary, [source, s_stream], table)

demo.launch()
