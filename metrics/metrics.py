from abc import ABC, abstractmethod

import math
from typing import Optional
import torch


class Metric(ABC):

    @abstractmethod
    def _calculate_metric(
        self,
        prediction_3d: torch.Tensor,
        groundtruth_3d: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def calculate_metric(
        self,
        prediction_3d: torch.Tensor,
        groundtruth_3d: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        error = self._calculate_metric(prediction_3d, groundtruth_3d)
        match reduction:
            case "none":
                return error
            case "mean":
                return error.nanmean()
            case "sum":
                return error.sum()

    def metric_at_timesteps(
        self,
        prediction_3d: torch.Tensor,
        groundtruth_3d: torch.Tensor,
        timesteps_in_ms: Optional[list[int]] = None,
        sampling_rate: int = 50,
        reduction: str = "mean"
    ) -> torch.Tensor:
        if timesteps_in_ms is not None:
            ms_per_index = 1000 / sampling_rate

            indices = []
            for timestep in timesteps_in_ms:
                i = math.floor(timestep / ms_per_index)
                # clip index into valid range
                i = min(i, prediction_3d.shape[-3])
                i = max(i, 0)

                indices.append(i)
        else:
            indices = range(1, prediction_3d.shape[-3] + 1)

        values = [
            self.calculate_metric(
                prediction_3d=prediction_3d[..., index - 1, :, :],
                groundtruth_3d=groundtruth_3d[..., index - 1, :, :],
                reduction=reduction
            ) for index in indices
        ]

        return values

    def metric_by_joint(
        self,
        prediction_3d: torch.Tensor,
        groundtruth_3d: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        return [
            self.calculate_metric(
                prediction_3d[..., joint_index, :],
                groundtruth_3d[..., joint_index, :],
                reduction=reduction
            ) for joint_index in range(prediction_3d.shape[-2])
        ]
