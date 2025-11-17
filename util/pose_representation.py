from enum import Enum

from data import DatasetType
from data.pose_representation import (
    PoseRepresentation,
    CenterScale,
    CenterJDScale,
    RotationMatrices
)

from util.data import get_pose_linkage


class RepresentationType(Enum):
    CenterScaleIndividual = "center-scale-individual"
    CenterJDScaleIndividual = "center-joint-distance-scale-individual"
    CenterScale = "center-scale"
    CenterScaleGtTraj = "center-scale-gt-traj"
    CenterJDScale = "center-joint-distance-scale"
    CenterJDScaleGtTraj = "center-joint-distance-scal-gt-traj"
    CenterLastJDScale = "center-last-pose-joint-distance-scale"
    RotationMatricesRelative = "rotation-matrices-relative"
    RotationMatricesAbsolute = "rotation-matrices-absolute"


def get_pose_representation(
    representation_type: RepresentationType,
    dataset_type: DatasetType | str
) -> PoseRepresentation:

    if isinstance(dataset_type, str):
        dataset_type = DatasetType(dataset_type)

    pose_linkage = get_pose_linkage(dataset_type)
    match representation_type:
        case RepresentationType.RotationMatricesRelative:
            return RotationMatrices(
                pose_linkage=pose_linkage,
                relative_orientations=True
            )
        case RepresentationType.RotationMatricesAbsolute:
            return RotationMatrices(
                pose_linkage=pose_linkage,
                relative_orientations=False
            )
        case RepresentationType.CenterScaleIndividual:
            return CenterScale(
                root_index=pose_linkage.root_index,
                centering_strategy="individual",
                scaling_factor=0.001,
                use_groundtruth_trajectory=False
            )
        case RepresentationType.CenterScale | RepresentationType.CenterScaleGtTraj:
            gt_traj = representation_type == RepresentationType.CenterScaleGtTraj
            print("here")
            print("gt_traj", gt_traj)
            return CenterScale(
                root_index=pose_linkage.root_index,
                centering_strategy="first",
                scaling_factor=0.001,
                use_groundtruth_trajectory=gt_traj
            )
        case (
            RepresentationType.CenterJDScale |
            RepresentationType.CenterJDScaleGtTraj |
            RepresentationType.CenterJDScaleIndividual |
            RepresentationType.CenterLastJDScale
        ):
            if representation_type == RepresentationType.CenterJDScaleGtTraj:
                gt_traj = True
            else:
                gt_traj = False

            match dataset_type:
                case DatasetType.HUMAN36M:
                    joint_a_index = 0
                    joint_b_index = 11
                    target_distance = 0.25
                case DatasetType.HA4M:
                    joint_a_index = 1
                    joint_b_index = 2
                    target_distance = 0.25
                case DatasetType.AMASS:
                    joint_a_index = 1
                    joint_b_index = 2
                    target_distance = 0.5
                case _:
                    raise NotImplementedError(
                        "No set of parameters defined for the CenterJDScale " +
                        f"pose representation for the given dataset {dataset_type}"
                    )

            # if RepresentationType.CenterJDScaleIndividual:
            if representation_type == RepresentationType.CenterJDScaleIndividual:
                return CenterJDScale(
                    root_index=pose_linkage.root_index,
                    centering_strategy="individual",
                    scale_joint_a_index=joint_a_index,
                    scale_joint_b_index=joint_b_index,
                    target_distance=target_distance,
                    use_groundtruth_trajectory=False
                )
            elif representation_type == RepresentationType.CenterJDScale:
                return CenterJDScale(
                    root_index=pose_linkage.root_index,
                    centering_strategy="first",
                    scale_joint_a_index=joint_a_index,
                    scale_joint_b_index=joint_b_index,
                    target_distance=target_distance,
                    use_groundtruth_trajectory=gt_traj
                )
            elif representation_type == RepresentationType.CenterLastJDScale:
                return CenterJDScale(
                    root_index=pose_linkage.root_index,
                    centering_strategy="last",
                    scale_joint_a_index=joint_a_index,
                    scale_joint_b_index=joint_b_index,
                    target_distance=target_distance,
                    use_groundtruth_trajectory=gt_traj
                )
            
