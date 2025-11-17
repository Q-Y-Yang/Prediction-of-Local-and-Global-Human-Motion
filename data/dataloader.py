from typing import Optional

from tqdm import tqdm

import math
import numpy as np

import torch
from torch.utils.data import Dataset

from data.data import MotionSequenceData
from data.pose_representation import PoseRepresentation
from data.joints_masking import random_joint_mask, structured_missing_joints, temporal_occlusion

class MotionSequenceDataset(Dataset):
    data: MotionSequenceData
    pose_representation: PoseRepresentation

    def __init__(
        self,
        data: MotionSequenceData,
        pose_representation: PoseRepresentation,
        history_seconds: float,
        future_seconds: float,
        stride_seconds: Optional[int] = 0.5,
        hist_remove_occluded: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.data = data
        self._n_joints = self.data.get_n_joints()
        self._joint_dim = self.data.get_joint_dim()

        self.pose_representation = pose_representation

        self._history_seconds = history_seconds
        self._future_seconds = future_seconds

        self.n_timesteps_history = math.ceil(
            self._history_seconds * data.sampling_rate
        )
        self.n_timesteps_future = math.ceil(
            self._future_seconds * data.sampling_rate
        )
        self.n_timesteps_full_sequence = (
            self.n_timesteps_history + self.n_timesteps_future
        )

        if stride_seconds is not None:
            self.stride_timesteps = int(
                stride_seconds * self.data.sampling_rate)
        else:
            self.stride_timesteps = 1

        self._build_sequences_index()

        self.hist_remove_occluded = hist_remove_occluded

        # if masking joints for missing joints training
        self.masking = kwargs.get("masking", False)
        self.mask_prob = kwargs.get("mask_prob", 0.05)

        # if random offset in the window [start_index, start_index + stride_timesteps]
        self.random_offset = kwargs.get("random_offset", True)


        for sequence_index in tqdm(
            range(self.data.get_n_sequences()),
            desc="Preprocessing Sequences"
        ):
            sequence = self.data.get_sequence(sequence_index)
            self.pose_representation.preprocess_sequence(sequence)

    def _build_sequences_index(self):
        indices = []
        for sequence_index in tqdm(
            range(self.data.get_n_sequences()),
            desc="Indexing Dataset Sequences"
        ):
            poses = self.data.get_sequence(sequence_index).poses
            start_indices = range(
                0, len(poses) -
                self.n_timesteps_full_sequence, self.stride_timesteps
            )
            for start_index in start_indices:
                indices.append((sequence_index, start_index))

        self.indices = indices

    @property
    def n_joints(self) -> int:
        return self._n_joints

    @property
    def joint_dim(self) -> int:
        return self._joint_dim

    @property
    def history_seconds(self) -> float:
        return self._history_seconds

    @history_seconds.setter
    def history_seconds(self, history_seconds: float):
        self.n_timesteps_history = math.ceil(
            history_seconds * self.data.sampling_rate
        )
        self.n_timesteps_full_sequence = (
            self.n_timesteps_history + self.n_timesteps_future
        )
        self._build_sequences_index()
        self._history_seconds = history_seconds

    @property
    def future_seconds(self) -> float:
        return self.future_seconds

    @future_seconds.setter
    def future_seconds(self, future_seconds: float):
        self.n_timesteps_future = math.ceil(
            future_seconds * self.data.sampling_rate
        )
        self.n_timesteps_full_sequence = (
            self.n_timesteps_history + self.n_timesteps_future
        )
        self._build_sequences_index()
        self._future_seconds = future_seconds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sequence_index, start_index = self.indices[index]

        # select sequence
        sequence = self.data.get_sequence(sequence_index)
        action_type = sequence.action.value
        frames = sequence.frames

        # sample start index in stride span
        if self.random_offset:
            history_start_index = np.random.randint(
                start_index,
                min(
                    start_index + self.stride_timesteps,
                    len(frames) - self.n_timesteps_full_sequence
                )
            )
        else:
            history_start_index = start_index

        sequence = frames[
            history_start_index:history_start_index + self.n_timesteps_full_sequence
        ]
        history = sequence[:self.n_timesteps_history]
        future = sequence[self.n_timesteps_history:]
        
        # apply pose representation
        (
            history, noisy, future_groundtruth,
            history_backtransform, future_backtransform
        ) = self.pose_representation.from_3d_coordinates(
            history=history, groundtruth=future
        )

        if self.hist_remove_occluded:
            joint_confidences = torch.stack([
                frame.additional_data.get("confidences")
                if frame.additional_data.get("confidences") is not None
                else torch.full([*frame.pose.shape[:-1], 1], dtype=torch.float32)
                for frame in sequence[:self.n_timesteps_history]
            ], dim=0)

            # mark joint data where confidence is lower than 0.6 as NaN
            history = torch.where(joint_confidences > 0.6, history, torch.nan)

                    # random and structured masking for missing joints
        if self.masking:
            poses_3d_random_joint = random_joint_mask(history, self.mask_prob)
            poses_3d_structured_missing, _ = structured_missing_joints(poses_3d_random_joint, self.mask_prob)
            history = temporal_occlusion(poses_3d_structured_missing, self.mask_prob)
                    
        return (
            history,
            noisy,
            future_groundtruth,
            history_backtransform,
            future_backtransform,
            action_type
        )
