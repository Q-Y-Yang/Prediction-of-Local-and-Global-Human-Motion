from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import torch

from data import PoseLinkage


def plot_metric_by_joint(
    metric_values: list[float],
    linkage: PoseLinkage,
    title: Optional[str] = None,
    metric_name: Optional[str] = None
) -> go.Figure:
    """Heatmap plot of metric values by joint.

    Arguments:
        metric_values -- Values to plot. Length should match number of joints.
        linkage -- PoseLinkage.

    Keyword Arguments:
        title -- If supplied the figure title will be set accordingly. (default: {None})
        metric_name -- If supplied it will be noted in the plot. (default: {None})

    Returns:
        go.Figure -- Figure containing the plot.
    """
    bodyparts = linkage.bodyparts
    max_joints_per_row = max([len(joints) for joints in bodyparts.values()])

    heatmap = []
    for joints_indices in bodyparts.values():
        row = [
            error for joint_index, error in enumerate(metric_values)
            if joint_index in joints_indices
        ]
        # pad row to max length with nan values
        row = row + (max_joints_per_row - len(row)) * [torch.nan]

        heatmap.append(row)

    fig = px.imshow(
        heatmap,
        y=list(bodyparts.keys()),
        title=title,
        text_auto=".2f",
        height=600,
        width=800
    )
    fig.update_layout(
        legend={"title": metric_name}
    )
    fig.update_xaxes(showticklabels=False)

    return fig


def plot_metric_by_ms(
    metric_values: list[float],
    sampling_rate: int,
    title: Optional[str] = None,
    metric_name: Optional[str] = None
) -> go.Figure:
    """Line plot of metric values by milliseconds

    Arguments:
        metric_values -- Values to plot. Length should match number of joints.
        sampling_rate -- Sampling rate of the data in fps.

    Keyword Arguments:
        title -- If supplied the figure title will be set accordingly. (default: {None})
        metric_name -- If supplied it will be noted in the plot. (default: {None})

    Returns:
        go.Figure -- Figure containing the plot.
    """
    fig = px.line(
        y=metric_values,
        x=[
            f"{(i + 1) * (1000 / sampling_rate):.0f}"
            for i in range(len(metric_values))
        ],
        text=[
            f"{value:.2f}" if i % 2 == 0 else torch.nan
            for i, value in enumerate(metric_values)
        ],
        markers=True,
        title=title,
        width=1200,
        height=500
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(yaxis_title=metric_name, xaxis_title="ms")

    return fig
