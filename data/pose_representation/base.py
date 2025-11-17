from abc import ABC, abstractmethod

import torch

from data.data import Frame, MotionSequence


class PoseRepresentation(ABC):

    @property
    @abstractmethod
    def joint_feature_dim(self) -> int:
        """ The number of dimensions of a joint feature vector of a pose
        represented by this pose representation.
        """
        pass

    @abstractmethod
    def preprocess_sequence(self, sequence: MotionSequence) -> None:
        """ Used to pre-process timestep independent transformations across 
        the motion sequences of a dataset. The transformations can be saved
        in the frames of the MotionSequence under frame.pose.
        Information that is required to transform the poses back to the original
        3D Pose Representation should be stored under frame.backtransformation_data

        Arguments:
            sequence -- MotionSequence
        """
        pass

    @abstractmethod
    def from_3d_coordinates(
        self,
        history: list[Frame],
        groundtruth: list[Frame]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Used to generate the training sample from history and groundtruth slices
        of a pre-processed MotionSequence.

        Arguments:
            history -- frames of the observed history
            groundtruth -- frames of the future

        Returns:
            history_transformed -- torch.Tensor
            noisy_transformed -- torch.Tensor
            groundtruth_transformed -- torch.Tensor
            history_backtransform_data -- torch.Tensor
            future_backtransform_data -- torch.Tensor
        """
        pass

    @abstractmethod
    def to_3d_coordinates(
        self,
        sequence_transformed: torch.Tensor,
        backtransform_data: torch.Tensor
    ) -> torch.Tensor:
        """ Used for the inverse transformation of a transformed sequence
        to 3D coordinates.

        Arguments:
            sequence_transformed -- sequence previously transformed by representation
            backtransform_data -- backtransform data from self.from_3d_coordinates(...)

        Returns:
            sequence_3d -- sequence backtransformed into 3D coordinates
        """
        pass
