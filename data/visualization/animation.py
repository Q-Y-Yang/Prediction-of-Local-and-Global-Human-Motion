from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Optional
import os

import torch
from PIL import Image

from data import DatasetType, PoseLinkage
from data.visualization.pose import plot_pose


def _create_pose_sequence_images(input_batch: tuple) -> list[Image.Image]:
    """Creates a list of images of pose plots (see visualization.pose.plot_pose). 
    The function is intended for multiprocessing and therefore only expects one
    tuple as input, which contains all arguments.

    Arguments:
        input_batch -- Tuple containing the following elements:
        - sequence: torch.Tensor - sequence of poses
        - linkage -- If supplied links will be drawn between joints (default: {None})
        - title -- If supplied the figure title will be modified. (default: {None})
        - minimum -- tuple (x, y, z) with the minimum for each axis. If not supplied 
        axis limits will not be modified (default: {None})
        - maximum -- tuple (x, y, z) with the maximum for each axis. If not supplied 
        axis limits will not be modified (default: {None})

    Returns:
        List of PIL Image objects of the individual pose plots for the given sequence.
    """
    sequence, linkage, title, minimum, maximum, dataset_type = input_batch

    images = []
    # pass pose_fig to function, so that it can be reused across poses
    pose_fig = None
    for pose in sequence:
        pose_fig = plot_pose(
            pose, fig=pose_fig, linkage=linkage, dataset_type=dataset_type,
            title=title, minimum=minimum, maximum=maximum
        )
        img_bytes = pose_fig.to_image(format="png")
        images.append(Image.open(BytesIO(img_bytes)))
        # reset figure data
        pose_fig.data = []

    return images


def create_prediction_vs_groundtruth_animation(
    history: torch.Tensor,
    prediction: torch.Tensor,
    groundtruth: torch.Tensor,
    linkage: PoseLinkage,
    save_filepath: str,
    fps: int,
    dataset_type: Optional[DatasetType] = None,
) -> None:
    """Creates a .gif file with an animation of the predicted and groundtruth 3D pose
    sequence next to each other. The animation is a sequence of individual pose plots
    (see visualization.pose.plot_pose).

    Arguments:
        history -- observed history of 3D poses (shape: [TIMESTEPS, N_JOINTS, 3])
        prediction -- predicted future of 3D poses (shape: [TIMESTEPS, N_JOINTS, 3])
        groundtruth -- grountruth future of 3D poses (shape: [TIMESTEPS, N_JOINTS, 3])
        linkage -- _description_
        save_filepath -- Filepath where the .gif file will be saved to.
        fps -- Frames per seconds for the animation. To create a realtime animation, 
        this should equal the sampling rate of the data.

    Keyword Arguments:
        dataset_type -- Used to set the scene camera. If none is given, a default 
        camera position will be set. (default: {None})
    """
    # calculate the max and min x, y and z values across all poses
    minmax = torch.cat([history, prediction, groundtruth], dim=0).view(-1, 3)
    x_min, x_max = (
        minmax[:, 0].nan_to_num(torch.inf).min(),
        minmax[:, 0].nan_to_num(-torch.inf).max()
    )
    y_min, y_max = (
        minmax[:, 1].nan_to_num(torch.inf).min(),
        minmax[:, 1].nan_to_num(-torch.inf).max()
    )
    z_min, z_max = (
        minmax[:, 2].nan_to_num(torch.inf).min(),
        minmax[:, 2].nan_to_num(-torch.inf).max()
    )
    minimum = [x_min, y_min, z_min]
    maximum = [x_max, y_max, z_max]

    multiprocessing_batch = [
        (history, linkage, "History", minimum, maximum, dataset_type),
        (prediction, linkage, "Prediction", minimum, maximum, dataset_type),
        (groundtruth, linkage, "Groundtruth", minimum, maximum, dataset_type)
    ]

    with ProcessPoolExecutor() as executor:
        # create the animation frames for each sequence
        images_history, images_prediction, images_groundtruth = list(
            executor.map(
                _create_pose_sequence_images, multiprocessing_batch
            )
        )

    images_left = [*images_groundtruth, *images_history]
    images_right = [*images_prediction, *images_history]

    # compose the two image sequences next to each other
    images = []
    for image_l, image_r in zip(images_left, images_right):
        width_l, height_l = image_l.size
        width_r, height_r = image_r.size

        image_composition = Image.new(
            'RGB', (width_l + width_r, max(height_r, height_l))
        )

        image_composition.paste(image_l, (0, 0))
        image_composition.paste(image_r, (width_l, 0))

        images.append(image_composition)

    # save image sequence as gif
    images[0].save(
        save_filepath, append_images=images[1:], save_all=True, loop=0,
        duration=1 / fps, optimize=True
    )
