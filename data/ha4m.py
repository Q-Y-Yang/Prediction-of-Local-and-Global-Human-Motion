from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import os
import glob

import numpy as np
import pandas as pd

import torch

from data.data import ActionType, Frame, MotionSequence, PoseLinkage, MotionSequenceData

ha4m_pose_linkage = PoseLinkage(
    root_index=0,
    links=[
        # hip
        (0, 18), (0, 22),
        # left leg
        (18, 19), (19, 20), (20, 21),
        # right leg
        (22, 23), (23, 24), (24, 25),
        # spine
        (0, 1), (1, 2), (2, 3),
        (2, 4), (2, 11),
        # head
        (3, 26),
        # left arm
        (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (7, 10),
        # right arm
        (11, 12), (12, 13), (13, 14),
        (14, 17), (14, 15), (15, 16)
    ]
)
ha4m_pose_linkage.add_bodypart_information(
    core_indices=[0, 1, 2, 3, 4, 11, 26],
    left_arm_indices=[5, 6, 7, 8, 9, 10],
    right_arm_indices=[12, 13, 14, 15, 16, 17],
    left_leg_indices=[18, 19, 20, 21],
    right_leg_indices=[22, 23, 24, 25]
)


class HA4MActionType(ActionType):
    Carrier = 1
    GearBearrings = 2
    PlanetGears = 3
    CarrierShaft = 4
    SunShaft = 5
    SunGear = 6
    SunGearBearring = 7
    RingBear = 8
    Block2on1 = 9
    Cover = 10
    Screws = 11
    Finish = 12


class HA4MData(MotionSequenceData):
    data_dir: str

    df: pd.DataFrame
    sequences: list[MotionSequence]

    pose_linkage = ha4m_pose_linkage
    action_types = HA4MActionType

    frames_per_second = 30
    sampling_rate: int

    train_subjects = [
        1, 2, 3, 4, 5, 7, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21,
        23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41
    ]

    def __init__(
        self,
        data_dir: str,
        train_valid_full: str = "train",
        sampling_rate: int = 30
    ) -> None:
        assert (train_valid_full in ["train", "valid", "full"])
        assert (sampling_rate in [10, 30])

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate

        self.sequences = []

        match train_valid_full:
            case "train":
                subjects = self.train_subjects
            case "valid":
                subjects = [
                    i for i in range(1, 42) if i not in self.train_subjects
                ]
            case _:
                subjects = list(range(1, 42))

        self.load_pose_data(
            subjects=subjects,
            sampling_rate=sampling_rate
        )

    def get_n_joints(self) -> int:
        return 27

    def get_joint_dim(self) -> int:
        return 3

    def process_frame(self, filepath: str):
        df = pd.read_csv(filepath, sep="\t", header=0)

        frame_id = int(
            os.path.basename(filepath)
            .split("_")[0]
            .replace("FrameID", "")
        )

        return frame_id, df

    def preprocess_sequence(self, filepath: str):
        save_filepath = os.path.join(filepath, "sequence_data.pkl")
        if os.path.isfile(save_filepath):
            return

        sequence_id = filepath.split(os.sep)[-1]
        subject = int(sequence_id[3:6])
        video_id = int(sequence_id[7:])

        frame_paths = glob.glob(
            os.path.join(filepath, "Skeletons", "*", "*.txt")
        )

        if "000099302712" in frame_paths[0]:
            camera_id = "000099302712"
        elif "000124702712" in frame_paths[0]:
            camera_id = "000124702712"
        else:
            camera_id = "-"

        frames = [
            self.process_frame(frame_path)
            for frame_path in frame_paths
        ]

        frame_records = [
            {
                "frame_id": frame_id,
                "pose_3d": frame[["X", "Y", "Z"]].to_numpy(),
                "pose_quaternion": frame[["Qw", "Qx", "Qy", "Qz"]].to_numpy(),
                "joint_confidences": frame["Confidance"].to_numpy()

            }
            for frame_id, frame in frames
        ]

        df_sequence = pd.DataFrame.from_records(frame_records)
        df_sequence["sequence_id"] = sequence_id
        df_sequence["video_id"] = video_id
        df_sequence["camera_id"] = camera_id
        df_sequence["subject"] = subject

        label_filepath = os.path.join(filepath, "Labels.txt")
        if os.path.isfile(label_filepath):
            labels = pd.read_csv(
                label_filepath, sep=" ", header=None,
                names=["frame", "0", "1"]
            )

            df_sequence["label_1"] = labels["0"]
            df_sequence["label_2"] = labels["1"]

        # check whether poses have uniform shape
        torch.tensor(np.array(
            df_sequence["pose_3d"].to_list()
        ))

        # save dataframe so that in the future it can be loaded faster
        df_sequence.to_pickle(save_filepath)

    def load_sequence(
        self,
        filepath: str,
        sampling_rate: int,
    ) -> tuple[MotionSequence, pd.DataFrame]:
        sequence_id = filepath.split(os.sep)[-1]
        subject = sequence_id[3:6]

        save_filepath = os.path.join(filepath, "sequence_data.pkl")
        if not os.path.isfile(save_filepath):
            self.preprocess_sequence(filepath=filepath)

        df_sequence = pd.read_pickle(save_filepath)

        skip_frames = int(self.frames_per_second / sampling_rate)

        poses_3d = torch.tensor(np.array(
            df_sequence["pose_3d"].to_list()
        ), dtype=torch.float32)

        joint_confidences = torch.tensor(np.array(
            df_sequence["joint_confidences"].to_list()
        ), dtype=torch.float32).unsqueeze(-1)
        joint_confidences = joint_confidences[::skip_frames]

        poses_3d = poses_3d[::skip_frames]
        # remove extra face keypoints
        poses_3d = poses_3d[:, :27, :]
        joint_confidences = joint_confidences[:, :27, :]

        frames = [
            Frame(pose=pose, additional_data={"confidences": confidences})
            for pose, confidences in zip(poses_3d, joint_confidences)
        ]

        sequence = MotionSequence(
            subject=subject, action=HA4MActionType.Finish, frames=frames,
        )

        return sequence, df_sequence

    def load_pose_data(
        self,
        subjects: list[int],
        sampling_rate: int = 30,
        n_workers: int = 4
    ):
        sessionpaths = glob.glob(os.path.join(self.data_dir, "IDU*"))
        sessionpaths = [
            sessionpath for sessionpath in sessionpaths
            if int(sessionpath.split(os.sep)[-1][3:6]) in subjects
        ]
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.preprocess_sequence, sessionpath): sessionpath
                    for sessionpath in sessionpaths
                }

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Loading HA4M Data"):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing file {futures[future]}: {e}")
        else:
            for sessionpath in tqdm(sessionpaths):
                self.preprocess_sequence(sessionpath)

        self.df = pd.DataFrame([])
        for sessionpath in sessionpaths:
            sequence, df_sequence = self.load_sequence(filepath=sessionpath,
                                                       sampling_rate=sampling_rate)

            self.sequences.append(sequence)
            self.df = pd.concat([self.df, df_sequence], ignore_index=True)

    def get_n_sequences(self) -> int:
        return len(self.sequences)

    def get_sequence(self, index: int) -> MotionSequence:
        return self.sequences[index]
