from enum import Enum

import torch

from data.data import Frame, MotionSequence
from data.pose_representation.base import PoseRepresentation


class CenteringStrategy(Enum):
    CENTER_INDIVIDUALLY = "individual"
    FIRST_HISTORY_POSE = "first"
    LAST_HISTORY_POSE = "last"
    MEAN_CENTER_JOINT_POSITION = "mean"


class CenterScale(PoseRepresentation):
    def __init__(
        self,
        root_index: int,
        scaling_factor: float,
        centering_strategy: str,
        use_groundtruth_trajectory: bool
    ) -> None:
        self.centering_strategy = CenteringStrategy(centering_strategy)
        self.root_index = root_index

        self.scaling_factor = torch.tensor([[[scaling_factor]]])

        self.use_groundtruth_trajectory = use_groundtruth_trajectory

    @property
    def joint_feature_dim(self) -> int:
        return 3

    def preprocess_sequence(self, sequence: MotionSequence) -> None:
        return None

    def from_3d_coordinates(
        self,
        history: list[Frame],
        groundtruth: list[Frame]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_3d = torch.stack(
            [frame.pose for frame in history], dim=0
        )
        groundtruth_3d = torch.stack(
            [frame.pose for frame in groundtruth], dim=0
        )

        match self.centering_strategy:
            case CenteringStrategy.FIRST_HISTORY_POSE:
                center_position = history_3d[0, self.root_index, None]
                center_position = center_position.unsqueeze(0)
            case CenteringStrategy.LAST_HISTORY_POSE:
                center_position = history_3d[-1, self.root_index, None]
                center_position = center_position.unsqueeze(0)
            case CenteringStrategy.MEAN_CENTER_JOINT_POSITION:
                # center sequences with respect to mean position
                # of center joint in history
                center_position = torch.mean(history_3d[:, 0, None], dim=0)
                center_position = center_position.unsqueeze(0)
            case CenteringStrategy.CENTER_INDIVIDUALLY:
                history_center_pos = history_3d[:, self.root_index]
                history_center_pos = history_center_pos.unsqueeze(1)
                groundtruth_center_pos = groundtruth_3d[:, self.root_index]
                groundtruth_center_pos = groundtruth_center_pos.unsqueeze(1)

        if self.centering_strategy == CenteringStrategy.CENTER_INDIVIDUALLY:
            history_centered = history_3d - history_center_pos
            groundtruth_centered = groundtruth_3d - groundtruth_center_pos

            history_backtransform = torch.cat(
                (
                    history_center_pos,
                    self.scaling_factor.repeat(history_3d.shape[0], 1, 1)
                ), dim=-1
            )
            future_backtransform = torch.cat(
                (
                    groundtruth_center_pos,
                    self.scaling_factor.repeat(groundtruth_3d.shape[0], 1, 1)
                ), dim=-1
            )
        else:
            history_centered = history_3d - center_position
            groundtruth_centered = groundtruth_3d - center_position

            backtransform_data = torch.cat(
                (center_position, self.scaling_factor), dim=-1
            )
            history_backtransform = backtransform_data
            future_backtransform = backtransform_data

        history_scaled = history_centered * self.scaling_factor
        groundtruth_scaled = groundtruth_centered * self.scaling_factor
        # noisy is last frame of history repeated
        noisy_scaled = history_scaled[-1].expand_as(
            groundtruth_scaled
        ).clone()

        if self.use_groundtruth_trajectory:
            noisy_scaled[:, self.root_index, :] = (
                groundtruth_scaled[:, self.root_index, :]
            )

        return (
            history_scaled,
            noisy_scaled,
            groundtruth_scaled,
            history_backtransform,
            future_backtransform
        )

    def to_3d_coordinates(
        self,
        sequence_transformed: torch.Tensor,
        backtransform_data: torch.Tensor
    ) -> torch.Tensor:
        scaling_factor = backtransform_data[..., 3]
        scaling_factor = scaling_factor.unsqueeze(-1)

        center_position = backtransform_data[..., :3]

        pos_centered = sequence_transformed / scaling_factor

        pos_3d = pos_centered + center_position

        return pos_3d


class CenterJDScale(PoseRepresentation):
    def __init__(
        self,
        root_index: int,
        centering_strategy: str,
        scale_joint_a_index: int,
        scale_joint_b_index: int,
        target_distance: float,
        use_groundtruth_trajectory: bool
    ) -> None:
        self.centering_strategy = CenteringStrategy(centering_strategy)
        self.root_index = root_index

        self.scale_joint_a_index = scale_joint_a_index
        self.scale_joint_b_index = scale_joint_b_index
        self.target_distance = torch.tensor([[[target_distance]]])

        self.use_groundtruth_trajectory = use_groundtruth_trajectory

    @property
    def joint_feature_dim(self) -> int:
        return 3

    def preprocess_sequence(self, sequence: MotionSequence) -> None:
        return None

    def from_3d_coordinates(
        self,
        history: list[Frame],
        groundtruth: list[Frame]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_3d = torch.stack(
            [frame.pose for frame in history], dim=0
        )
        groundtruth_3d = torch.stack(
            [frame.pose for frame in groundtruth], dim=0
        )

        match self.centering_strategy:
            case CenteringStrategy.FIRST_HISTORY_POSE:
                center_position = history_3d[0, self.root_index, None]
                center_position = center_position.unsqueeze(0)
            case CenteringStrategy.LAST_HISTORY_POSE:
                center_position = history_3d[-1, self.root_index, None]
                center_position = center_position.unsqueeze(0)
            case CenteringStrategy.MEAN_CENTER_JOINT_POSITION:
                # center sequences with respect to mean position
                # of center joint in history
                center_position = torch.mean(history_3d[:, 0, None], dim=0)
                center_position = center_position.unsqueeze(0)
            case CenteringStrategy.CENTER_INDIVIDUALLY:
                history_center_pos = history_3d[:, self.root_index]
                history_center_pos = history_center_pos.unsqueeze(1)
                groundtruth_center_pos = groundtruth_3d[:, self.root_index]
                groundtruth_center_pos = groundtruth_center_pos.unsqueeze(1)

        if self.centering_strategy == CenteringStrategy.CENTER_INDIVIDUALLY:
            history_centered = history_3d - history_center_pos
            groundtruth_centered = groundtruth_3d - groundtruth_center_pos
        else:
            history_centered = history_3d - center_position
            groundtruth_centered = groundtruth_3d - center_position

        # calculate the mean distance between the scale joint indices
        mean_scale_joint_distance = torch.mean(
            torch.nn.functional.pairwise_distance(
                history_centered[:, self.scale_joint_a_index, :],
                history_centered[:, self.scale_joint_b_index, :]
            )
        )
        scaling_factor = self.target_distance / mean_scale_joint_distance

        history_scaled = history_centered * scaling_factor
        groundtruth_scaled = groundtruth_centered * scaling_factor
        # noisy is last frame of history repeated
        noisy_scaled = history_scaled[-1].expand_as(
            groundtruth_scaled
        ).clone()

        if self.centering_strategy == CenteringStrategy.CENTER_INDIVIDUALLY:
            history_backtransform = torch.cat(
                (
                    history_center_pos,
                    scaling_factor.repeat(history_3d.shape[0], 1, 1)
                ), dim=-1
            )
            future_backtransform = torch.cat(
                (
                    groundtruth_center_pos,
                    scaling_factor.repeat(groundtruth_3d.shape[0], 1, 1)
                ), dim=-1
            )
        else:
            backtransform_data = torch.cat(
                (center_position, scaling_factor), dim=-1
            )
            history_backtransform = backtransform_data
            future_backtransform = backtransform_data

        if self.use_groundtruth_trajectory:
            noisy_scaled[:, self.root_index, :] = (
                groundtruth_scaled[:, self.root_index, :]
            )

        return (
            history_scaled,
            noisy_scaled,
            groundtruth_scaled,
            history_backtransform,
            future_backtransform
        )

    def to_3d_coordinates(
        self,
        sequence_transformed: torch.Tensor,
        backtransform_data: torch.Tensor
    ) -> torch.Tensor:
        scaling_factor = backtransform_data[..., 3]
        scaling_factor = scaling_factor.unsqueeze(-1)

        center_position = backtransform_data[..., :3]

        pos_centered = sequence_transformed / scaling_factor

        pos_3d = pos_centered + center_position

        return pos_3d
