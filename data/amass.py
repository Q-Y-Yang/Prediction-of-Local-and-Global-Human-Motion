from typing import Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import os
import glob

import random
import numpy as np
import pandas as pd

import torch
import smplx

from data.data import ActionType, Frame, MotionSequence, PoseLinkage, MotionSequenceData


def get_amass_pose_linkage(simple_model: bool = False) -> PoseLinkage:
    if simple_model:
        amass_pose_linkage = PoseLinkage(
            root_index=2,
            links=[
                # hip
                (2, 0), (2, 1),

                # left leg
                (0, 3), (3, 6), (6, 8),
                # right leg
                (1, 4), (4, 7), (7, 9),

                # spine
                (2, 5), (5, 10),
                (5, 12), (5, 11),
                (10, 13),
                # head
                (13, 20), (13, 21),

                # left arm
                (11, 14), (14, 16), (16, 18),
                # left hand
                (18, 22), (18, 23), (18, 24),

                # right arm
                (12, 15), (15, 17), (17, 19),
                # right hand
                (19, 25), (19, 26), (19, 27),
            ]
        )

        amass_pose_linkage.add_bodypart_information(
            core_indices=[
                # spine
                2, 5, 10, 11,
                # shoulders
                12, 13,
                # head
                20, 21,
            ],
            left_arm_indices=[
                # arm
                14, 16, 18,
                # hand: index - middle - thumb
                22, 23, 24
            ],
            right_arm_indices=[
                # arm
                15, 17, 19,
                # hand: index - middle - thumb
                25, 26, 27
            ],
            left_leg_indices=[0, 3, 6, 8],
            right_leg_indices=[1, 4, 7, 9]
        )
    else:
        amass_pose_linkage = PoseLinkage(
            root_index=0,
            links=[
                # hip
                (0, 1), (0, 2), (0, 3),  # (3, 1), (3, 2),
                # left leg
                (1, 4), (4, 7), (7, 10),
                # right leg
                (2, 5), (5, 8), (8, 11),
                # spine
                (3, 6), (6, 9),
                (9, 12), (9, 13), (9, 14),
                # head
                (12, 15),  # head
                (15, 22),  # jaw
                (15, 23), (15, 24),  # eyes
                # left arm
                (13, 16), (16, 18), (18, 20),  # arm
                # left fingers
                (20, 25), (25, 26), (26, 27),  # index
                (20, 28), (28, 29), (29, 30),  # middle
                (20, 31), (31, 32), (32, 33),  # pinky
                (20, 34), (34, 35), (35, 36),  # ring
                (20, 37), (37, 38), (38, 39),  # thumb
                # right arm
                (14, 17), (17, 19), (19, 21),  # arm
                # right fingers
                (21, 40), (40, 41), (41, 42),  # index
                (21, 43), (43, 44), (44, 45),  # middle
                (21, 46), (46, 47), (47, 48),  # pinky
                (21, 49), (49, 50), (50, 51),  # ring
                (21, 52), (52, 53), (53, 54),  # thumb
            ]
        )

        amass_pose_linkage.add_bodypart_information(
            core_indices=[
                0, 3, 6, 9, 12, 13, 14,  # spine
                15, 22, 23, 24,  # head
            ],
            left_arm_indices=[
                16, 18, 20,  # arm
                25, 26, 27,  # index
                28, 29, 30,  # middle
                31, 32, 33,  # pinky
                34, 35, 36,  # ring
                37, 38, 39,  # thumb
            ],
            right_arm_indices=[
                17, 19, 21,  # arm
                40, 41, 42,  # index
                43, 44, 45,  # middle
                46, 47, 48,  # pinky
                49, 50, 51,  # ring
                52, 53, 54,  # thumb
            ],
            left_leg_indices=[1, 4, 7, 10],
            right_leg_indices=[2, 5, 8, 11]
        )

    return amass_pose_linkage


class AMASSActionType(ActionType):
    ACCAD = "ACCAD"
    BML_MOVI = "BMLmovi"
    BML_RUB = "BMLrub"
    CMU = "CMU"
    CNRS = "CNRS"
    DANCE_DB = "DanceDB"
    DFAUST = "DFaust"
    EKUT = "EKUT"
    EYES_JAPAN = "Eyes_Japan_Dataset"
    GRAB = "GRAB"
    HDM05 = "HDM05"
    HUMAN4D = "HUMAN4D"
    HUMAN_EVA = "HumanEva"
    KIT = "KIT"
    MOSH = "MoSh"
    POSE_PRIOR = "PosePrior"
    SFU = "SFU"
    SOMA = "SOMA"
    SSM = "SSM"
    TCD_HANDS = "TCDHands"
    TOTAL_CAPTURE = "TotalCapture"
    TRANSITIONS = "Transitions"
    WEIZMANN = "WEIZMANN"


class AMASSData(MotionSequenceData):
    data_dir: str

    sequences: list[MotionSequence]

    pose_linkage: PoseLinkage
    action_types = AMASSActionType

    frames_per_second = 60
    sampling_rate: int

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        train_valid_full: str = "train",
        sampling_rate: int = 30,
        simple_model: bool = False,
        preload_sequences: bool = True,
        preprocess: bool = False
    ) -> None:
        assert (train_valid_full in ["train", "valid", "test", "full"])
        assert (sampling_rate in [10, 20, 30, 60])

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.sampling_rate = sampling_rate
        self.datasets_mixed = False

        self.is_simple_model = simple_model
        self.pose_linkage = get_amass_pose_linkage(
            simple_model=self.is_simple_model
        )

        if preprocess:
            print("Pre-Processing AMASS datasets")
            self.smplx_models = {}
            for gender in ["male", "female", "neutral"]:
                model = smplx.create(
                    self.model_dir,
                    model_type="smplx",
                    num_betas=16,
                    num_expression_coeffs=10,
                    gender=gender,
                    ext="npz",
                    use_pca=False,
                    batch_size=64,
                )
                model.eval()
                self.smplx_models[gender] = model

            datasets = [e.value for e in AMASSActionType]

            for dataset in datasets:
                self.preprocess_dataset(dataset=dataset)

        glob_string = os.path.join(
            self.data_dir, "*", "*", "processed", "*.pkl"
        )
        sequence_filepaths = glob.glob(glob_string)
        if not self.datasets_mixed:
            amass_split = {
                "train": [
                    "ACCAD", "BMLmovi", "BMLrub", "CMU", "EKUT",
                    "Eyes_Japan_Dataset", "KIT", "PosePrior",
                    "TCDHands", "TotalCapture", "CNRS", "DanceDB",
                    "DFaust", "GRAB", "WEIZMANN", "SSM"
                ],
                "valid": [
                    "HDM05", "HumanEva", "SFU", "SOMA", "HUMAN4D", "Transitions"
                ],
                "test": ["MoSh"]
            }

            match train_valid_full:
                case "train":
                    datasets = amass_split["train"]
                case "valid":
                    datasets = amass_split["valid"]
                case "test":
                    datasets = amass_split["test"]
                case _:
                    datasets = [
                        *amass_split["train"],
                        *amass_split["valid"]
                    ]

            sequence_filepaths = list(filter(
                lambda f: f.split(os.sep)[-4] in datasets,
                sequence_filepaths
            ))
        else:
            # Split into train (80%), valid (10%), test (10%)
            total_count = len(sequence_filepaths)
            random.seed(42)
            random.shuffle(sequence_filepaths)

            test_split = int(total_count * 0.1)
            valid_split = int(total_count * 0.2)

            test_sequences = sequence_filepaths[:test_split]
            val_sequences = sequence_filepaths[test_split:valid_split]
            train_sequences = sequence_filepaths[valid_split:]

            random.seed(None)

            match train_valid_full:
                case "train":
                    sequence_filepaths = train_sequences
                case "valid":
                    sequence_filepaths = val_sequences
                case "test":
                    sequence_filepaths = test_sequences
                case _:
                    sequence_filepaths = sequence_filepaths  # full set

            print(f"Split into {len(train_sequences)} train, {len(val_sequences)} valid, {len(test_sequences)} test sequences.")

        self.sequence_filepaths = sequence_filepaths

        self.preload_sequences = preload_sequences
        if self.preload_sequences:
            self.sequences = []
            for sequence_filepath in tqdm(
                self.sequence_filepaths, desc="Preloading Sequences"
            ):
                sequence = self.load_sequence(
                    sequence_filepath=sequence_filepath,
                    simple_model=self.is_simple_model,
                    sampling_rate=sampling_rate,
                )
                self.sequences.append(sequence)

    def get_n_joints(self) -> int:
        return 55

    def get_joint_dim(self) -> int:
        return 3

    def get_body_model(self, gender: str, batch_size: int) -> smplx.SMPLX:
        self.smplx_models[gender].batch_size = batch_size
        return self.smplx_models[gender]

    def _preprocess_sequence(self, sequence_filepath: str) -> Optional[pd.DataFrame]:
        dataset = sequence_filepath.split(os.sep)[-3]
        subject_id = sequence_filepath.split(os.sep)[-2]
        sequence_id = sequence_filepath.split(os.sep)[-1].replace(".npz", "")

        # load/save data so that body model only needs to be computed once
        sequence_folder = os.sep.join(sequence_filepath.split(os.sep)[:-1])
        save_filepath = os.path.join(
            sequence_folder, "processed", f"{sequence_id}.pkl"
        )
        if os.path.isfile(save_filepath):
            df = pd.read_pickle(save_filepath)
            return df

        # mmap_mode = "r" loads only data that is accessed
        # f.e. by calling sequence_data["pose_body"]
        sequence_data = np.load(
            sequence_filepath,
            allow_pickle=True,
            mmap_mode="r"
        )

        # some files dont have motion data
        if "pose_body" not in sequence_data.keys():
            sequence_data.close()
            return None

        gender = str(sequence_data["gender"])

        # load pose parameters
        pose_data = {
            key: torch.tensor(sequence_data[key], dtype=torch.float32)
            for key in [
                "root_orient", "trans", "betas",
                "pose_body", "pose_hand", "pose_jaw", "pose_eye"
            ]
        }
        sequence_length = pose_data["pose_body"].shape[0]
        pose_data["betas"] = pose_data["betas"].repeat(sequence_length, 1)

        sequence_data.close()

        with torch.no_grad():
            # process sequence in batches
            batch_size = 2048
            results = []
            for start in range(0, sequence_length, batch_size):
                end = min(start + batch_size, sequence_length)
                model = self.get_body_model(
                    gender=gender, batch_size=end-start
                )
                output = model(
                    global_orient=pose_data["root_orient"][start:end],
                    transl=pose_data["trans"][start:end],
                    betas=pose_data["betas"][start:end],
                    body_pose=pose_data["pose_body"][start:end],
                    leye_pose=pose_data["pose_eye"][start:end, :3],
                    reye_pose=pose_data["pose_eye"][start:end, 3:],
                    jaw_pose=pose_data["pose_jaw"][start:end],
                    left_hand_pose=pose_data["pose_hand"][start:end, :45],
                    right_hand_pose=pose_data["pose_hand"][start:end, 45:],
                    expression=torch.zeros([end-start, 10])
                )
                results.append(output.joints)

        poses_3d = torch.cat(results, dim=0)
        # resample from 120 -> 60 fps
        poses_3d = poses_3d[::2]
        # remove unused face details
        poses_3d = poses_3d[:, :55, :]

        df_sequence = pd.DataFrame({"pose_3d": list(poses_3d.numpy())})
        df_sequence["subject_id"] = subject_id
        df_sequence["sequence_id"] = sequence_id
        df_sequence["dataset"] = dataset
        df_sequence["gender"] = gender

        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        df_sequence.to_pickle(save_filepath)

        return df_sequence

    def preprocess_dataset(
        self,
        dataset: str,
        n_workers: int = 1
    ) -> None:
        # do not use to many workers at once -> easily clogs up ram
        df_dataset = pd.DataFrame([])

        glob_string = os.path.join(self.data_dir, dataset, "*", "*.npz")
        sequence_filepaths = glob.glob(glob_string)

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self._preprocess_sequence, filepath): filepath
                    for filepath in sequence_filepaths
                }

                for future in tqdm(
                    as_completed(futures), desc=f"{dataset}", total=len(futures)
                ):
                    try:
                        df_sequence = future.result()
                        if df_sequence is not None:
                            df_dataset = pd.concat(
                                [df_dataset, df_sequence], ignore_index=True
                            )
                    except Exception as e:
                        print(f"Error processing file {futures[future]}: {e}")
        else:
            for sequence_filepath in tqdm(sequence_filepaths, desc=f"{dataset}"):
                df_sequence = self._preprocess_sequence(
                    sequence_filepath=sequence_filepath
                )
                df_dataset = pd.concat(
                    [df_dataset, df_sequence], ignore_index=True
                )

        df_dataset.to_pickle(
            os.path.join(self.data_dir, dataset, "dataset.pkl")
        )

    def load_sequence(
        self,
        sequence_filepath: str,
        simple_model: bool,
        sampling_rate: int
    ) -> MotionSequence:
        df_sequence = pd.read_pickle(sequence_filepath)
        poses_3d = torch.tensor(np.array(
            df_sequence["pose_3d"].to_list()
        ), dtype=torch.float32) * 1000

        # resample data to sampling rate
        skip_frames = int(self.frames_per_second / sampling_rate)
        poses_3d = poses_3d[::skip_frames]

        if simple_model:
            joint_selection = [
                j for j in range(25)
                if j not in [0, 9, 22]
            ]

            joint_selection = joint_selection + [
                # left hand
                25, 37, 28,
                # right hand
                40, 52, 43,
            ]

            poses_3d = poses_3d[:, joint_selection]

        frames = [Frame(pose=pose) for pose in poses_3d]

        sequence = MotionSequence(
            subject=df_sequence["subject_id"][0],
            action=AMASSActionType(df_sequence["dataset"][0]),
            frames=frames
        )

        return sequence

    def get_n_sequences(self) -> int:
        if self.preload_sequences:
            return len(self.sequences)
        else:
            return len(self.sequence_filepaths)

    def get_sequence(self, index: str) -> MotionSequence:
        if self.preload_sequences:
            return self.sequences[index]
        else:
            sequence = self.load_sequence(
                self.sequence_filepaths[index],
                simple_model=self.is_simple_model,
                sampling_rate=self.sampling_rate,
            )

            return sequence
