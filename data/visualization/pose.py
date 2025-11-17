from typing import Optional

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from data import DatasetType, PoseLinkage


def _get_scene_camera_for_dataset(dataset_type: Optional[DatasetType] = None) -> dict:
    """Returns a dictionary with the scene camera angle for a given dataset type."""

    match dataset_type:
        case DatasetType.HUMAN36M:
            return dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        case DatasetType.HA4M:
            return dict(eye=dict(x=0.0, y=-0.4, z=-2.25))
        case _:
            return dict(eye=dict(x=1.5, y=-1.5, z=1.0))


def apply_figure_styling_to_3d_plot(
    fig: go.Figure,
    dataset_type: Optional[DatasetType] = None,
    title: Optional[str] = None,
    minimum: Optional[tuple[float]] = None,
    maximum: Optional[tuple[float]] = None,
    show_legend: bool = True,
) -> None:
    """Applies a default styling to a plotly 3D-Scatter plot.

    Arguments:
        fig -- The figure which will be modified

    Keyword Arguments:
        dataset_type -- Used to set the scene camera. If none is given, a default 
        camera position will be set. (default: {None})
        title -- If supplied the figure title will be modified. (default: {None})
        minimum -- tuple (x, y, z) with the minimum for each axis. If not supplied 
        axis limits will not be modified (default: {None})
        maximum -- tuple (x, y, z) with the maximum for each axis. If not supplied 
        axis limits will not be modified (default: {None})
        show_legend -- (default: {True})
    """
    fig.update_layout(
        width=600,
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=show_legend
    )

    # set title
    if title is not None:
        fig.update_layout(
            title=dict(text=title, x=0.5, y=0.9),
        )

    # set axis limits if minimum and maximum are given
    if minimum is not None and maximum is not None:
        xrange = [minimum[0], maximum[0]]
        yrange = [minimum[1], maximum[1]]
        zrange = [minimum[2], maximum[2]]
    else:
        xrange, yrange, zrange = None, None, None

    # general axis properties
    axis = dict(
        backgroundcolor="rgba(0, 0, 0, 0.025)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgb(0, 0, 0)',
        zeroline=False,
        showticklabels=False
    )

    # set the axes and camera position
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", **axis, range=xrange),
            yaxis=dict(title="y", **axis, range=yrange),
            zaxis=dict(title="z", **axis, range=zrange),
            aspectmode="cube",
        ),
        scene_camera=_get_scene_camera_for_dataset(dataset_type)
    )


def _add_joint_links_to_fig(
    pose: torch.Tensor,
    fig: go.Figure,
    linkage: PoseLinkage,
    alpha: float = 1.0
) -> None:
    """Adds 3D links to the given figure according to the skeleton in pose linkage.

    Arguments:
        pose -- Pose for which the links should be added. The pose should
        have the shape [N_JOINTS, 3].
        fig -- The figure where joint links will be added to.
        linkage -- Linkage that is used as reference for joint links.

    Keyword Arguments:
        alpha -- Opacity of the drawn links. (default: {1.0})
    """
    red = "255, 0, 0"
    green = "0, 100, 0"
    blue = "0, 0, 140"

    body_part_color = {
        "core": blue,
        "left_arm": green, "left_leg": green,
        "right_arm": red, "right_leg": red
    }
    for bodypart, joint_indices in linkage.bodyparts.items():
        color = body_part_color[bodypart]
        links = [link for link in linkage.links if link[1] in joint_indices]

        x, y, z = [], [], []
        for joint_a, joint_b in links:
            # link is made up of a path a->b(->None)
            # None marks the end of the path
            x.extend([pose[joint_a, 0], pose[joint_b, 0], None])
            y.extend([pose[joint_a, 1], pose[joint_b, 1], None])
            z.extend([pose[joint_a, 2], pose[joint_b, 2], None])

        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines', line=dict(width=3),
                marker=dict(color=f'rgba({color}, {alpha})'),
                name=bodypart, legendgroup=bodypart
            )
        )


def plot_pose(
    pose: torch.Tensor,
    fig: Optional[go.Figure] = None,
    linkage: Optional[PoseLinkage] = None,
    dataset_type: Optional[DatasetType] = None,
    title: Optional[str] = None,
    minimum: Optional[tuple[float]] = None,
    maximum: Optional[tuple[float]] = None,
    show_legend: bool = False,
    alpha: float = 1.0
) -> go.Figure:
    """Plots the supplied pose as a 3D scatter plot. If linkage is given, then
    3D lines are added to the plot that represent the links between joints.

    Arguments:
        pose -- Pose that should be plotted (shape: [N_JOINTS, 3])

    Keyword Arguments:
        fig -- If none is given a new one is created (default: {None})
        linkage -- If supplied links will be drawn between joints (default: {None})
        dataset_type -- Used to set the scene camera. If none is given, a default 
        camera position will be set. (default: {None})
        title -- If supplied the figure title will be modified. (default: {None})
        minimum -- tuple (x, y, z) with the minimum for each axis. If not supplied 
        axis limits will not be modified (default: {None})
        maximum -- tuple (x, y, z) with the maximum for each axis. If not supplied 
        axis limits will not be modified (default: {None})
        show_legend -- (default: {False})
        alpha -- Opacity of the joints and links (default: {1.0})

    Returns:
        go.Figure -- Figure containing the plot.
    """
    xs = pose[:, 0]
    ys = pose[:, 1]
    zs = pose[:, 2]

    # create figure if none is given
    if fig is None:
        fig = go.Figure()
        apply_figure_styling_to_3d_plot(
            fig, title=title, show_legend=show_legend,
            minimum=minimum, maximum=maximum,
            dataset_type=dataset_type
        )

    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            name="joints", legendgroup="joints",
            mode='markers',
            marker=dict(size=4, color=f'rgba(0, 0, 0, {alpha})'),
            text=[
                f"Joint-{joint_index}"
                for joint_index in range(len(xs))
            ],
        )
    )

    # plot links between joints if pose linkage is given
    if linkage is not None:
        _add_joint_links_to_fig(pose, fig, linkage, alpha=alpha)

    return fig


def plot_pose_sequence(
    sequence: torch.Tensor,
    linkage: Optional[PoseLinkage] = None,
    dataset_type: Optional[DatasetType] = None,
    title: Optional[str] = None,
    minimum: Optional[tuple[float]] = None,
    maximum: Optional[tuple[float]] = None
) -> go.Figure:
    """Plots the supplied poses together in one 3D scatter plot. (see plot_pose)

    Arguments:
        sequence -- Poses (shape: [TIMESTEP, N_JOINTS, 3])

    Keyword Arguments:
        linkage -- If supplied links will be drawn between joints (default: {None})
        dataset_type -- Used to set the scene camera. If none is given, a default 
        camera position will be set. (default: {None})
        dataset_type -- Used to set the scene camera. If none is given, a default 
        camera position will be set. (default: {None})
        title -- If supplied the figure title will be modified. (default: {None})
        minimum -- tuple (x, y, z) with the minimum for each axis. If not supplied 
        axis limits will not be modified (default: {None})
        maximum -- tuple (x, y, z) with the maximum for each axis. If not supplied 
        axis limits will not be modified (default: {None})

    Returns:
        go.Figure -- Figure containing the plot.
    """
    fig = go.Figure()

    for i, pose in enumerate(sequence):
        if i % 5 == 0 or i == len(sequence) - 1:
            alpha = 1.0
        else:
            alpha = 0.3
        
        alpha = 1.0

        plot_pose(pose, fig=fig, linkage=linkage, alpha=alpha)

    apply_figure_styling_to_3d_plot(
        fig=fig,
        dataset_type=dataset_type,
        title=title,
        minimum=minimum,
        maximum=maximum,
        show_legend=False
    )

    return fig


def plot_prediction_vs_groundtruth_pose_sequence(
    prediction: torch.Tensor,
    groundtruth: torch.Tensor,
    linkage: Optional[PoseLinkage] = None,
    dataset_type: Optional[DatasetType] = None
) -> go.Figure:
    """Creates a figure with two subplots for the prediction and groundtruth
    pose sequence. For subplots see plot_pose_sequence.

    Arguments:
        prediction -- Predicted pose (shape: [N_JOINTS, 3])
        groundtruth -- Groundtruth pose (shape: [N_JOINTS, 3])

    Keyword Arguments:
        linkage -- If supplied links will be drawn between joints (default: {None})
        dataset_type -- Used to set the scene camera. If none is given, a default 
        camera position will be set. (default: {None})

    Returns:
        go.Figure -- Figure containing the plot.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=("Groundtruth", "Prediction")
    )

    fig_1 = plot_pose_sequence(groundtruth, linkage=linkage)
    for trace in fig_1.data:
        trace.update(scene="scene1")
        fig.add_trace(trace, row=1, col=1)

    fig_2 = plot_pose_sequence(prediction, linkage=linkage)
    for trace in fig_2.data:
        trace.update(scene="scene2")
        fig.add_trace(trace, row=1, col=2)

    camera = _get_scene_camera_for_dataset(dataset_type)
    axis = dict(
        backgroundcolor="rgba(0, 0, 0, 0.05)",
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgb(0, 0, 0)',
        zeroline=False,
        showticklabels=False
    )

    fig.update_layout(
        scene1=dict(
            xaxis=dict(title="x", **axis),
            yaxis=dict(title="y", **axis),
            zaxis=dict(title="z", **axis),
            aspectmode="cube",
            camera=camera
        ),
        scene2=dict(
            xaxis=dict(title="x", **axis),
            yaxis=dict(title="y", **axis),
            zaxis=dict(title="z", **axis),
            aspectmode="cube",
            camera=camera
        ),
        width=1200, height=750,
        showlegend=False
    )

    return fig
