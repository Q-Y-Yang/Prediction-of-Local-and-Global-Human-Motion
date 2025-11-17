import os
import glob

import torch

import numpy as np
import pandas as pd

from spacepy import pycdf

from data.data import ActionType, Frame, MotionSequence, PoseLinkage, MotionSequenceData

human36m_pose_linkage = PoseLinkage(
    root_index=0,
    links=[
        # back
        (0, 11), (11, 12), (12, 13),
        # head
        (13, 14),
        # left arm
        (12, 15), (15, 16), (16, 17),
        # left hand
        (17, 18), (17, 19),
        # right arm
        (12, 20), (20, 21), (21, 22),
        # right hand
        (22, 23), (22, 24),
        # left leg
        (0, 6), (6, 7), (7, 8),
        (8, 9), (9, 10),
        # right leg
        (0, 1), (1, 2), (2, 3),
        (3, 4), (4, 5),
    ]
)
human36m_pose_linkage.add_bodypart_information(
    core_indices=[0, 1, 6, 11, 12, 13, 14, 15, 20],
    left_arm_indices=[16, 17, 18, 19],
    right_arm_indices=[21, 22, 23, 24],
    left_leg_indices=[7, 8, 9, 10],
    right_leg_indices=[2, 3, 4, 5]
)


class Human36MActionType(ActionType):
    Discussion = 0
    Smoking = 1
    Phoning = 2
    SittingDown = 3
    Walking = 4
    Sitting = 5
    Waiting = 6
    Eating = 7
    Directions = 8
    WalkTogether = 9
    Greeting = 10
    Photo = 11
    Posing = 12
    WalkDog = 13
    Purchases = 14
    WalkingDog = 15
    TakingPhoto = 16


class Human36MData(MotionSequenceData):
    pose_linkage = human36m_pose_linkage
    action_types = Human36MActionType

    sequences: list[MotionSequence]

    data_dir: str

    frames_per_second = 50
    sampling_rate: int

    df: pd.DataFrame

    def __init__(
        self,
        data_dir: str,
        train_valid_full: str = "train",
        sampling_rate: int = 50
    ) -> None:
        assert (train_valid_full in ["train", "valid", "full"])
        assert (sampling_rate in [50, 25, 10])

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate

        self.sequences = []

        match train_valid_full:
            case "train":
                subjects = [1, 5, 6, 7, 8]
            case "valid":
                subjects = [9, 11]
            case "full":
                subjects = [1, 5, 6, 7, 8, 9, 11]
            case _:
                subjects = []

        self.load_pose_data(subjects=subjects, sampling_rate=sampling_rate)

    def get_n_joints(self) -> int:
        for seq in self.sequences:
            if len(seq.poses) > 0:
                return seq.poses[0].shape[0]

        # return 0 if no sample was found
        return 0

    def get_joint_dim(self) -> int:
        for seq in self.sequences:
            if len(seq.poses) > 0:
                return seq.poses[0].shape[1]

        # return 0 if no sample was found
        return 0

    def load_pose_data(self, subjects: list[int], sampling_rate: int = 50) -> None:
        globpath = os.path.join(
            self.data_dir, "poses", "S*", "D3_Positions", "*.cdf"
        )

        skip_frames = int(self.frames_per_second / sampling_rate)

        self.df = pd.DataFrame([])
        for filepath in glob.glob(globpath):
            subject = int(filepath.split(os.sep)[-3].removeprefix("S"))

            # skip unwanted subjects
            if subject not in subjects:
                continue

            # action string including version suffix f.e. "Walking 1"
            action_id = filepath.split(os.sep)[-1].removesuffix(".cdf")
            # remove number indicators from when one subject has
            # done a type of action multiple times
            action = "".join(s for s in action_id if not s.isdigit())
            # remove whitespaces
            action = action.replace(" ", "")

            # load pose data from .cdf file
            data = pycdf.CDF(filepath)
            raw_pose_data = np.array(data["Pose"])
            # reshape sequences
            # from (timeframe, x0y0z0x1y1z1x2..) where x0 = x coordinate of joint 0
            # to   (timeframe, joint, xyz)
            poses_3d = raw_pose_data.reshape(raw_pose_data.shape[1], 32, 3)

            # remove duplicate joint data
            duplicate_joint_indices = [31, 28, 24, 23, 20, 16, 11]
            poses_3d = np.delete(poses_3d, duplicate_joint_indices, axis=1)

            # resample to desired sampling rate
            poses_3d = poses_3d[::skip_frames]

            # add sequence to dataframe
            df_sequence = pd.DataFrame(
                [[subject, action_id, pose] for pose in poses_3d],
                columns=["subject", "action", "pose_3d"],
            )
            
            poses_3d = torch.tensor(poses_3d)

            sequence = MotionSequence(
                frames=[Frame(pose=pose) for pose in poses_3d],
                subject=str(subject), action=Human36MActionType[action]
            )
            # print(sequence.frames[9].pose)  # Limit decimals for compactness
            self.df = pd.concat([self.df, df_sequence], ignore_index=True)

            self.sequences.append(sequence)


    def get_n_sequences(self) -> int:
        return len(self.sequences)

    def get_sequence(self, index: int) -> MotionSequence:
        return self.sequences[index]
