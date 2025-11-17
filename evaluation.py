import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics
from data import DatasetType, MotionSequenceDataset
from data.pose_representation import PoseRepresentation
from data.visualization.animation import \
    create_prediction_vs_groundtruth_animation
from networks import NetworkType, load_model_snapshot
from util import (RepresentationType, get_dataset, get_pose_linkage,
                  get_pose_representation)

DEVICE = "cuda"


def get_filepath_for_best_training_snapshot(log_dir: str) -> Optional[str]:
    filepaths = glob.glob(f"{log_dir}/best*.model")
    if len(filepaths) == 0:
        return None
    filepath = filepaths[0]

    return filepath


def evaluate_training_run(
    log_dir: str,
    network_type: NetworkType,
    dataset_type: DatasetType,
    pose_representation: RepresentationType | PoseRepresentation,
    eval_metrics: list[metrics.Metric],
    history_seconds: float = 1,
    future_seconds: float = 1,
    sampling_rate: Optional[int] = None,
    training_with_occlusions: bool = False,
    num_prediction_visualizations: int = 3
) -> tuple[dict, dict]:
    weights_path = get_filepath_for_best_training_snapshot(log_dir)

    model = load_model_snapshot(network_type=network_type,
                                filepath=weights_path)
    if model is None:
        print("could not load model")
        return None
    model = model.to(DEVICE)

    if isinstance(pose_representation, RepresentationType):
        pose_representation = get_pose_representation(
            representation_type=pose_representation,
            dataset_type=dataset_type
        )

    valid_dataset = MotionSequenceDataset(
        data=get_dataset(
            dataset_type=dataset_type,
            train_valid_full="valid",
            sampling_rate=sampling_rate
        ),
        pose_representation=pose_representation,
        history_seconds=history_seconds,
        future_seconds=future_seconds,
        stride_seconds=None,
        hist_remove_occluded=training_with_occlusions
    )
    dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

    model.eval()

    results_df = {"metric": [], "action": [], "value": []}
    results_by_timesteps_df = {"metric": [], "action": [], "value": []}

    # predict all validation samples
    first_batch = True
    n_samples = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating Best Model")
        for batches in pbar:
            history = batches[0].to(DEVICE)
            noisy = batches[1].to(DEVICE)
            groundtruth = batches[2]
            future_backtransform = batches[4]
            action_types = batches[5]

            n_samples += history.shape[0]

            prediction = model(history, noisy)
            prediction = prediction.cpu()

            prediction_3d = pose_representation.to_3d_coordinates(
                prediction,
                future_backtransform
            )
            groundtruth_3d = pose_representation.to_3d_coordinates(
                groundtruth,
                future_backtransform
            )

            # calculate metrics
            for metric in eval_metrics:
                values = metric.calculate_metric(
                    prediction_3d=prediction_3d,
                    groundtruth_3d=groundtruth_3d,
                    reduction="none"
                )

                for value, action_type in zip(values, action_types):
                    results_df["metric"].append(str(metric))
                    results_df["action"].append(action_type)
                    results_df["value"].append(value)

                values_by_timesteps = metric.metric_at_timesteps(
                    prediction_3d=prediction_3d,
                    groundtruth_3d=groundtruth_3d,
                    sampling_rate=sampling_rate,
                    reduction="none"
                )

                values_by_timesteps = torch.tensor(np.array(
                    values_by_timesteps
                )).permute(1, 0, 2)

                for value, action in zip(values_by_timesteps, action_types):
                    results_by_timesteps_df["metric"].append(str(metric))
                    results_by_timesteps_df["action"].append(action)
                    results_by_timesteps_df["value"].append(value)

            if first_batch:
                first_batch = False
                history_backtransform = batches[3]
                history_3d = pose_representation.to_3d_coordinates(
                    history.cpu(),
                    history_backtransform
                )
                # visualize predictions
                for i in range(num_prediction_visualizations):
                    pbar.set_description("Creating Prediction Animations")
                    create_prediction_vs_groundtruth_animation(
                        history=history_3d[i].cpu(),
                        groundtruth=groundtruth_3d[i].cpu(),
                        prediction=prediction_3d[i].cpu(),
                        linkage=get_pose_linkage(dataset_type),
                        fps=sampling_rate,
                        save_filepath=os.path.join(
                            log_dir, "plots",
                            f"valid_prediction_best_epoch_{i}.gif"
                        ),
                        dataset_type=dataset_type
                    )
                    pbar.set_description("Evaluating Best Model")

    results_df = pd.DataFrame(results_df)
    results_by_timesteps_df = pd.DataFrame(results_by_timesteps_df)

    # calculate mean across all samples
    mean_metrics = {}
    mean_metrics_by_timesteps = {}
    for metric in eval_metrics:
        mean = results_df[results_df["metric"] == str(metric)]["value"].apply(
            lambda t: t.mean()
        ).mean()
        mean_metrics[str(metric)] = mean

        mean_by_timesteps = torch.tensor(np.array(
            results_by_timesteps_df[
                results_by_timesteps_df["metric"] == str(metric)]
            ["value"].to_list()
        )).mean(dim=0).mean(dim=-1)
        mean_metrics_by_timesteps[str(metric)] = mean_by_timesteps

    # write evaluation results
    with open(os.path.join(log_dir, "best_model_evaluation.txt"), "a") as f:
        f.write(f"Observation Window: {history_seconds}s\n")
        f.write("Mean Metrics:\n")
        for metric, mean in mean_metrics.items():
            f.write(f"{metric}:\n")
            f.write(f"{mean}\n")
        f.write("\nMean Metrics by Timestep:\n")
        for metric, mean_by_timesteps in mean_metrics_by_timesteps.items():
            f.write(f"{metric}:\n")
            f.write(f"{mean_by_timesteps}\n")

    return mean_metrics, mean_metrics_by_timesteps
