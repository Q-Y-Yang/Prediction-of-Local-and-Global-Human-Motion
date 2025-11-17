from typing import Optional

import torch

import networkx as nx

from enum import Enum
from abc import ABC, abstractmethod


class ActionType(Enum):
    pass


class DefaultAction(ActionType):
    DefaultAction = "NONE"


class Frame:

    pose: torch.Tensor

    # data that is needed to retransform the pose into 3D coordinates
    # should be set if pose is transformed by any pose representation
    # that needs it
    backtransformation_data: torch.Tensor

    # additonal data that might be used in the dataloader
    # f.e. indicating which joints are occluded
    additional_data: dict[str, torch.Tensor]

    def __init__(
        self,
        pose: torch.Tensor,
        backtransformation_data: Optional[torch.Tensor] = None,
        additional_data: Optional[dict[str, torch.Tensor]] = None
    ) -> None:
        self.pose = pose

        if backtransformation_data is None:
            self.backtransformation_data = torch.tensor([])
        else:
            self.backtransformation_data = backtransformation_data

        self.additional_data = additional_data


class MotionSequence:
    subject: str
    action: str
    frames: list[Frame]

    def __init__(
        self,
        subject: str,
        action: Optional[ActionType] = None,
        frames: Optional[list] = None
    ) -> None:
        self.subject = subject
        self.action = action

        if frames is not None:
            self.frames = frames
        else:
            self.frames = []

        if action is not None:
            self.action = action
        else:
            self.action = DefaultAction.DefaultAction

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def poses(self) -> list[torch.Tensor]:
        return [frame.pose for frame in self.frames]

    def __str__(self) -> str:
        s = f"S{self.subject}"
        s += f"-{self.action}"
        s += f"-{len(self.frames)}frames"

        return s


class PoseLinkage:

    root_index: int
    linkage_graph: nx.Graph

    bodyparts: Optional[dict[str, list[int]]]

    def __init__(self, links: list[tuple[int, int]], root_index: int) -> None:
        self.root_index = root_index

        graph = nx.DiGraph()

        nodes = set()
        for link in links:
            nodes.add(link[0])
            nodes.add(link[1])

        for node in sorted(list(nodes)):
            graph.add_node(node)

        for a, b in links:
            graph.add_edge(a, b)

        self.linkage_graph = graph

    def add_bodypart_information(
        self,
        core_indices: list[int],
        left_arm_indices: list[int],
        right_arm_indices: list[int],
        left_leg_indices: list[int],
        right_leg_indices: list[int],
    ):
        self.bodyparts = {
            "core": core_indices,
            "left_arm": left_arm_indices,
            "right_arm": right_arm_indices,
            "left_leg": left_leg_indices,
            "right_leg": right_leg_indices,
        }

    @property
    def n_joints(self) -> int:
        return len(self.linkage_graph.nodes)

    @property
    def joints(self) -> list[int]:
        return list(self.linkage_graph.nodes)

    @property
    def joints_in_topological_order(self) -> list[int]:
        return list(nx.topological_sort(self.linkage_graph))

    @property
    def links(self) -> list[tuple[int, int]]:
        return list(self.linkage_graph.edges)

    def parent_of(self, index: int) -> int:
        if index == self.root_index:
            return index

        for link in self.links:
            if link[1] == index:
                return link[0]

        return index

    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.linkage_graph).todense()


class MotionSequenceData(ABC):

    pose_linkage: PoseLinkage
    action_types: ActionType

    # fps of the raw data
    frames_per_second: int
    # data will be resampled to this target sampling rate
    sampling_rate: int

    @abstractmethod
    def get_n_joints(self) -> int:
        """Provides the number of joints of a pose from this dataset.

        Returns:
            int: the number of joints
        """
        pass

    @abstractmethod
    def get_joint_dim(self) -> int:
        """Provides the number of dimensions of a joint vector of a pose
        from this dataset.

        Returns:
            int: the number of joints
        pass
        """

    @abstractmethod
    def get_n_sequences(self) -> int:
        """Provides the total number of unique sequences in the dataset.

        Returns:
            int: the number of sequences in the dataset
        """
        pass

    @abstractmethod
    def get_sequence(self, index: int) -> MotionSequence:
        """Is used to access the sequences in the dataset.

        Arguments:
            index -- int: the index of the sequence based on the 
            total number of sequences.

        Returns:
            A unique MotionSequence that is associated with the given index.
        """
        pass
