from typing import Optional

from config import AMASSConfig, HA4MConfig, Human36MConfig

from data import DatasetType, PoseLinkage
from data.data import MotionSequenceData
from data.amass import AMASSData, get_amass_pose_linkage
from data.ha4m import HA4MData, ha4m_pose_linkage
from data.human36m import Human36MData, human36m_pose_linkage


def get_pose_linkage(dataset_type: DatasetType | str) -> PoseLinkage:
    if isinstance(dataset_type, str):
        dataset_type = DatasetType(dataset_type)

    match dataset_type:
        case DatasetType.HUMAN36M:
            return human36m_pose_linkage
        case DatasetType.HA4M:
            return ha4m_pose_linkage
        case DatasetType.AMASS:
            return get_amass_pose_linkage(simple_model=True)


def get_dataset(
    dataset_type: DatasetType | str,
    train_valid_full: str = "full",
    sampling_rate: Optional[int] = None
) -> MotionSequenceData:
    if isinstance(dataset_type, str):
        dataset_type = DatasetType(dataset_type)

    match dataset_type:
        case DatasetType.HUMAN36M:
            if sampling_rate is None:
                sampling_rate = 25
            return Human36MData(
                data_dir=Human36MConfig.DATA_DIR,
                train_valid_full=train_valid_full,
                sampling_rate=sampling_rate
            )
        case DatasetType.HA4M:
            if sampling_rate is None:
                sampling_rate = 30
            return HA4MData(
                data_dir=HA4MConfig.DATA_DIR,
                train_valid_full=train_valid_full,
                sampling_rate=sampling_rate
            )
        case DatasetType.AMASS:
            if sampling_rate is None:
                sampling_rate = 30
            return AMASSData(
                data_dir=AMASSConfig.DATA_DIR,
                model_dir=AMASSConfig.MODEL_DIR,
                train_valid_full=train_valid_full,
                sampling_rate=sampling_rate,
                preprocess=False,
                simple_model=True
            )
