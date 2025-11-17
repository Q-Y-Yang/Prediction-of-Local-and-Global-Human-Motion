from enum import Enum

from data.data import PoseLinkage
from data.dataloader import MotionSequenceDataset


class DatasetType(Enum):
    HUMAN36M = "human36m"
    HA4M = "ha4m"
    AMASS = "amass"


__all__ = [
    "PoseLinkage",
    "MotionSequenceDataset"
]
