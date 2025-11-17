import glob
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics

from data import DatasetType, MotionSequenceDataset, PoseLinkage
from data.pose_representation import PoseRepresentation, RotationMatrices
from data.visualization import (
    create_prediction_vs_groundtruth_animation,
    plot_metric_by_joint,
    plot_metric_by_ms,
    plot_prediction_vs_groundtruth_pose_sequence
)
from evaluation import evaluate_training_run
from loggers import MultiLogger, TensorboardLogger
from networks import (
    NetworkType,
    SpatioTemporalTransformer,
    GCNTemporalTransformer,
    BaselineTransformer,
    GraphInfluenceTransformer,
    load_model_snapshot
)
from util import (
    RepresentationType,
    get_dataset,
    get_pose_linkage,
    get_pose_representation
)
from loss import CosineLossLimb

import matplotlib.pyplot as plt
import gc

DEVICE = "cuda"

EPOCHS = 100
MODEL_SAVE_INTERVAL_IN_EPOCHS = 5

TRAINING_METRICS: list[metrics.Metric] = [
    metrics.MPJPE(),
]
METRIC_TIMESTEPS = [80, 160, 320, 400, 1000]


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        lower_is_better: bool = True
    ) -> None:
        self.patience = patience
        self.epochs_without_improvement = 0

        if lower_is_better:
            self.best_validation_score = torch.inf
        else:
            self.best_validation_score = - torch.inf

    def step(self, validation_score: torch.Tensor):
        if validation_score < self.best_validation_score:
            self.epochs_without_improvement = 0
            self.best_validation_score = validation_score
        else:
            self.epochs_without_improvement += 1

    @property
    def stop_now(self) -> bool:
        return self.epochs_without_improvement > self.patience


def create_datasets(
    dataset_type: DatasetType,
    pose_representation: PoseRepresentation,
    batch_size: int,
    history_seconds: float,
    future_seconds: float,
    sampling_rate: Optional[int] = None,
    training_with_occlusions: bool = False
):
    train = MotionSequenceDataset(
        data=get_dataset(
            dataset_type=dataset_type,
            train_valid_full="train",
            sampling_rate=sampling_rate
        ),
        pose_representation=pose_representation,
        history_seconds=history_seconds,
        future_seconds=future_seconds,
        stride_seconds=0.5,
        hist_remove_occluded=training_with_occlusions,
        masking=True,
        mask_prob=0.05
    )

    valid = MotionSequenceDataset(
        data=get_dataset(
            dataset_type=dataset_type,
            train_valid_full="valid",
            sampling_rate=sampling_rate
        ),
        pose_representation=pose_representation,
        history_seconds=history_seconds,
        future_seconds=future_seconds,
        hist_remove_occluded=training_with_occlusions,
        masking=True,
        mask_prob=0.05
    )

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader


def train(
    model: torch.nn.Module,
    train_dataloader,
    pose_representation: PoseRepresentation,
    optimizer: torch.optim.Optimizer,
    loss_functions: list[tuple[float, torch.nn.Module]],
    sampling_rate: int,
    logger,
    vary_observation_windows_range: Optional[tuple] = None,
):
    model.train()

    n_samples = 0
    train_losses = []
    train_metrics = defaultdict(list)
    train_metrics_by_timesteps = {
        str(metric): defaultdict(list)
        for metric in TRAINING_METRICS
    }
    for batches in train_dataloader:
        history = batches[0]
        if vary_observation_windows_range is not None:
            history_length = torch.distributions.uniform.Uniform(
                vary_observation_windows_range[0], vary_observation_windows_range[1]
            ).sample([1])
            history_length = torch.floor(history_length * sampling_rate).int()
            history = history[:, -history_length:, :, :]

        history = history.to(DEVICE)
        noisy = batches[1].to(DEVICE)
        groundtruth = batches[2].to(DEVICE)

        batch_size = history.shape[0]
        n_samples += batch_size

        optimizer.zero_grad()
        prediction_batch = model(history, noisy)

        # compute loss
        combined_loss = 0
        for weight, loss_function in loss_functions:
            loss = weight * loss_function(
                prediction_batch, groundtruth
            )
            combined_loss += loss

            logger.add_scalar(
                f"loss/{loss_function.__class__.__name__}", loss,
            )

        combined_loss.backward()
        optimizer.step()

        # logging
        logger.add_scalar("loss", combined_loss)
        train_losses.append(combined_loss * batch_size)

        with torch.no_grad():
            future_backtransform = batches[4].to(DEVICE)
            pred_3d = pose_representation.to_3d_coordinates(
                prediction_batch,
                future_backtransform
            )
            groundtruth_3d = pose_representation.to_3d_coordinates(
                groundtruth,
                future_backtransform
            )

            for metric in TRAINING_METRICS:
                error = metric.calculate_metric(
                    prediction_3d=pred_3d, groundtruth_3d=groundtruth_3d
                )
                logger.add_scalar(str(metric), error)
                train_metrics[str(metric)].append(
                    error * batch_size
                )

                error_by_timesteps = metric.metric_at_timesteps(
                    prediction_3d=pred_3d,
                    groundtruth_3d=groundtruth_3d,
                    timesteps_in_ms=METRIC_TIMESTEPS,
                    sampling_rate=sampling_rate
                )

                for ms, value in zip(METRIC_TIMESTEPS, error_by_timesteps):
                    train_metrics_by_timesteps[str(metric)][ms].append(
                        value * batch_size
                    )

        logger.step()

    optimizer.zero_grad()

    train_loss = logger.add_scalar_epoch(
        "loss", train_losses, n_samples=n_samples
    )

    epoch_mean_by_metric = {}
    for metric in TRAINING_METRICS:
        mean = logger.add_scalar_epoch(
            str(metric), train_metrics[str(metric)],
            n_samples=n_samples
        )

        epoch_mean_by_metric[str(metric)] = mean.item()

        for ms in METRIC_TIMESTEPS:
            logger.add_scalar_epoch(
                f"{metric}_MS/{ms}",
                train_metrics_by_timesteps[str(metric)][ms],
                n_samples=n_samples
            )

    return train_loss, epoch_mean_by_metric


def valid(
    model: torch.nn.Module,
    valid_dataloader,
    pose_representation: PoseRepresentation,
    loss_functions: list[tuple[float, torch.nn.Module]],
    sampling_rate: int,
    pose_linkage: PoseLinkage,
    logger,
    dataset_type: DatasetType
):
    model.eval()

    n_samples = 0
    valid_losses = []
    valid_metrics = defaultdict(list)
    valid_metrics_by_timesteps = {
        str(metric): defaultdict(list)
        for metric in TRAINING_METRICS
    }
    metrics_by_timesteps_plot = defaultdict(list)
    metric_by_joints_plot = defaultdict(list)
    with torch.no_grad():
        for batch_idx, batches in enumerate(valid_dataloader):
            hist_batch, noisy_batch, truth_batch = (
                batches[0].to(DEVICE),
                batches[1].to(DEVICE),
                batches[2].to(DEVICE)
            )

            n_samples += hist_batch.shape[0]

            prediction_batch = model(hist_batch, noisy_batch)

            # compute loss
            combined_loss = 0
            for weight, loss_function in loss_functions:
                loss = weight * loss_function(
                    prediction_batch, truth_batch
                )
                combined_loss += loss

            valid_losses.append(combined_loss * hist_batch.shape[0])

            future_backtransform = batches[4].to(DEVICE)
            pred_3d = pose_representation.to_3d_coordinates(
                prediction_batch,
                future_backtransform
            )
            groundtruth_3d = pose_representation.to_3d_coordinates(
                truth_batch,
                future_backtransform
            )

            # calculate metrics
            for metric in TRAINING_METRICS:
                error = metric.calculate_metric(
                    prediction_3d=pred_3d, groundtruth_3d=groundtruth_3d
                )
                valid_metrics[str(metric)].append(
                    error * hist_batch.shape[0]
                )

                error_by_timesteps = metric.metric_at_timesteps(
                    prediction_3d=pred_3d,
                    groundtruth_3d=groundtruth_3d,
                    timesteps_in_ms=METRIC_TIMESTEPS,
                    sampling_rate=sampling_rate
                )

                for ms, value in zip(METRIC_TIMESTEPS, error_by_timesteps):
                    valid_metrics_by_timesteps[str(metric)][ms].append(
                        value * hist_batch.shape[0]
                    )

                errors_by_timesteps = metric.metric_at_timesteps(
                    prediction_3d=pred_3d.cpu(),
                    groundtruth_3d=groundtruth_3d.cpu(),
                    sampling_rate=sampling_rate
                )
                metric_by_joint = metric.metric_by_joint(
                    prediction_3d=pred_3d.cpu(),
                    groundtruth_3d=groundtruth_3d.cpu()
                )
                metrics_by_timesteps_plot[str(metric)].append(
                    errors_by_timesteps
                )
                metric_by_joints_plot[str(metric)].append(metric_by_joint)

            # create result visualizations
            create_plot = logger.current_epoch % 2 == 0
            create_animation = logger.current_epoch % 5 == 0
            if create_plot or create_animation:
                pred_3d = pred_3d.cpu()
                groundtruth_3d = groundtruth_3d.cpu()

            # log pose sequence prediction plots every 5 epochs
            if create_plot and batch_idx == 0:
                fig = plot_prediction_vs_groundtruth_pose_sequence(
                    prediction=pred_3d[0], groundtruth=groundtruth_3d[0],
                    linkage=pose_linkage, dataset_type=dataset_type
                )
                logger.add_figure(
                    fig=fig, fig_name="prediction-vs-groundtruth"
                )
                plt.close(fig)

            if create_animation and batch_idx == 0:
                history = hist_batch.cpu()
                history_backtransform = batches[3].cpu()
                history_3d = pose_representation.to_3d_coordinates(
                    history, history_backtransform
                )
                for i in range(1):
                    create_prediction_vs_groundtruth_animation(
                        history=history_3d[i],
                        groundtruth=groundtruth_3d[i],
                        prediction=pred_3d[i],
                        linkage=pose_linkage,
                        save_filepath=os.path.join(
                            logger.log_dir, "plots",
                            f"valid_prediction_epoch_{logger.current_epoch}_{i}.gif"
                        ),
                        fps=sampling_rate,
                        dataset_type=dataset_type
                    )
                gc.collect()

    valid_loss = logger.add_scalar_epoch(
        "loss", valid_losses, n_samples=n_samples,
        valid=True
    )

    epoch_mean_by_metric = {}
    for metric in TRAINING_METRICS:
        mean = logger.add_scalar_epoch(
            str(metric), valid_metrics[str(metric)],
            n_samples=n_samples, valid=True
        )
        epoch_mean_by_metric[str(metric)] = mean.item()

        for ms in METRIC_TIMESTEPS:
            logger.add_scalar_epoch(
                f"{metric}_MS/{ms}",
                valid_metrics_by_timesteps[str(metric)][ms],
                n_samples=n_samples,
                valid=True
            )

        metric_values = metrics_by_timesteps_plot[str(metric)]
        metric_values = torch.tensor(metric_values).mean(dim=0)

        fig = plot_metric_by_ms(
            metric_values, sampling_rate=sampling_rate,
            title="Validation Prediction MPJPE by ms " +
            f"(Epoch {logger.current_epoch})",
            metric_name=str(metric)
        )
        logger.add_figure(fig=fig, fig_name=f"{metric}-by-ms")
        plt.close(fig)

        metric_values = metric_by_joints_plot[str(metric)]
        metric_values = torch.tensor(metric_values).mean(dim=0)
        fig = plot_metric_by_joint(
            metric_values, linkage=pose_linkage,
            title="Validation Position Error by Joint " +
            f"(Epoch {logger.current_epoch})",
            metric_name=str(metric)
        )
        logger.add_figure(fig=fig, fig_name=f"{metric}-by-joints")
        plt.close(fig)

        metrics_by_timesteps_plot.clear()
        metric_by_joints_plot.clear()
        torch.cuda.empty_cache()
        gc.collect()

    return valid_loss, epoch_mean_by_metric


def main(
    network_type: NetworkType,
    checkpoint_path: str,
    dataset_type: DatasetType,
    pose_representation: RepresentationType | PoseRepresentation,
    loss_functions: list[tuple[float, torch.nn.Module]],
    subfolder: Optional[str] = None,
    vary_observation_windows_range: Optional[tuple] = None,
    training_with_occlusions: bool = False,
    batch_size: int = 128,
    history_seconds: float = 1,
    future_seconds: float = 1,
    sampling_rate: Optional[int] = None,
    embedding_dim: int = 64,
    n_blocks: int = 3,
    num_heads: int = 8,
):
    if vary_observation_windows_range is not None:
        max_history_length = vary_observation_windows_range[1]

        if history_seconds != max_history_length:
            history_seconds = max_history_length

    if training_with_occlusions:
        if dataset_type != DatasetType.HA4M:
            raise NotImplementedError(
                "Training with Occlusions only implemented with HA4M dataset"
            )
        if network_type in [
            NetworkType.BASELINE_TRANSFORMER, NetworkType.GRAPH_INFLUENCE_TRANSFORMER
        ]:
            raise NotImplementedError(
                "Training with Occlusions not implemented for Baseline Transformer models"
            )

    pose_linkage = get_pose_linkage(dataset_type=dataset_type)

    if isinstance(pose_representation, RepresentationType):
        pose_representation = get_pose_representation(
            representation_type=pose_representation,
            dataset_type=dataset_type
        )

    train_dataloader, valid_dataloader = create_datasets(
        dataset_type=dataset_type,
        pose_representation=pose_representation,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        history_seconds=history_seconds,
        future_seconds=future_seconds
    )

    if sampling_rate is None:
        sampling_rate = train_dataloader.dataset.data.sampling_rate

    if isinstance(pose_representation, RotationMatrices):
        activation = "tanh"
    else:
        activation = "linear"

    match network_type:
        case NetworkType.BASELINE_TRANSFORMER:
            model = BaselineTransformer(
                n_joints=pose_linkage.n_joints,
                joint_feature_dim=pose_representation.joint_feature_dim,
                embedding_dim=embedding_dim,
                max_length=math.ceil(
                    sampling_rate * max(history_seconds, future_seconds)
                ),
                n_transformer_blocks=n_blocks,
                n_attention_heads=num_heads,
                output_activation=activation
            )
        case NetworkType.GRAPH_INFLUENCE_TRANSFORMER:
            model = GraphInfluenceTransformer(
                n_joints=pose_linkage.n_joints,
                joint_feature_dim=pose_representation.joint_feature_dim,
                pose_linkage=pose_linkage,
                embedding_dim=embedding_dim,
                max_length=math.ceil(
                    sampling_rate * max(history_seconds, future_seconds)
                ),
                k_hops=4,
                discount_factor=0.8,
                n_transformer_blocks=n_blocks,
                n_attention_heads=num_heads,
                output_activation=activation
            )
        case NetworkType.SPATIO_TEMPORAL:
            model = SpatioTemporalTransformer(
                n_joints=pose_linkage.n_joints,
                joint_feature_dim=pose_representation.joint_feature_dim,
                joint_embedding_dim=embedding_dim,
                max_length=math.ceil(
                    sampling_rate * max(history_seconds, future_seconds)
                ),
                n_attention_blocks=n_blocks,
                num_heads=num_heads,
                output_activation=activation
            )
        case NetworkType.GCN_TEMPORAL:
            model = GCNTemporalTransformer(
                n_joints=pose_linkage.n_joints,
                joint_feature_dim=pose_representation.joint_feature_dim,
                joint_embedding_dim=embedding_dim,
                max_length=math.ceil(
                    sampling_rate * max(history_seconds, future_seconds)
                ),
                n_attention_blocks=n_blocks,
                num_heads=num_heads,
                output_activation=activation
            )

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint_file = glob.glob(os.path.join(checkpoint_path, 'best_posenet_*.model'))[0]
        print(f"Loading checkpoint from {checkpoint_file}")
        model = load_model_snapshot(network_type=network_type,
                                filepath=checkpoint_file)
        
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params: {total_params}")
    print(f"trainable_params: {trainable_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=10e-4, weight_decay=10e-5
    )

    # declare hyperparameters for logging
    hyperparameters = {
        "network_type": network_type.value,
        "dataset": dataset_type.value,
        "representation": type(pose_representation).__name__,
        "loss_function": '-'.join([
            f"{loss_func.__class__.__name__}-{weight}"
            for weight, loss_func in loss_functions
        ]),
        "optimizer": str(optimizer),
        "n_blocks": n_blocks,
        "embedding_dim": embedding_dim,
        "sampling_rate": sampling_rate,
        "history_seconds": history_seconds,
        "future_seconds": future_seconds,
        "n_heads": num_heads,
        "batch_size": batch_size,
    }

    architecture_name = f"{network_type.value}_{type(pose_representation).__name__}"

    time = datetime.now()
    run_name = time.strftime('%y-%m-%d_%H-%M-%S')
    run_name += f"-{architecture_name}-{dataset_type.value}"
    run_name += f"-{hyperparameters['loss_function']}"

    if subfolder is not None:
        log_dir = os.path.join("logs", subfolder, run_name)
    else:
        log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=False)

    logger = MultiLogger(
        log_dir=log_dir,
        loggers=[
            TensorboardLogger(
                log_dir=log_dir, hyperparameters=hyperparameters
            ),
        ]
    )

    best_validation_loss = 111111
    early_stopping = EarlyStopping(patience=5, lower_is_better=True)

    for e in range(EPOCHS):
        train_loss, train_epoch_mean_by_metric = train(
            model=model,
            train_dataloader=tqdm(
                train_dataloader,
                desc=f"Train: Epoch {e+1}/{EPOCHS}",
                unit="batch"
            ),
            pose_representation=pose_representation,
            optimizer=optimizer,
            loss_functions=loss_functions,
            sampling_rate=sampling_rate,
            logger=logger,
            vary_observation_windows_range=vary_observation_windows_range
        )

        optimizer.zero_grad()

        valid_loss, valid_epoch_mean_by_metric = valid(
            model=model,
            valid_dataloader=tqdm(
                valid_dataloader,
                desc=f"Valid: Epoch {e+1}/{EPOCHS}",
                unit="batch"
            ),
            pose_representation=pose_representation,
            loss_functions=loss_functions,
            sampling_rate=sampling_rate,
            pose_linkage=pose_linkage,
            logger=logger,
            dataset_type=dataset_type
        )

        early_stopping.step(valid_loss)
        logger.step_epoch()

        epoch_end_string = f"       Epoch {e+1}/{EPOCHS}: "
        epoch_end_string += "; ".join([
            f"Train - Loss: {train_loss:.3f}",
            *[f"{m}: {v:.2f}"for m, v in train_epoch_mean_by_metric.items()],
            f"Valid - Loss: {valid_loss:.3f}",
            *[f"{m}: {v:.2f}"for m, v in valid_epoch_mean_by_metric.items()],
            f"Early Stopping {early_stopping.epochs_without_improvement}"
        ])
        print(epoch_end_string)

        # save best model
        if valid_loss < best_validation_loss:
            best_validation_loss = valid_loss
            # remove old best model snapshot
            for filepath in glob.glob(f"{log_dir}/best*.model"):
                os.remove(filepath)

            # save new best snapshot
            model.save_model(
                path=os.path.join(log_dir, f"best_posenet_{e+1}.model")
            )

        if (e+1) % MODEL_SAVE_INTERVAL_IN_EPOCHS == 0 or early_stopping.stop_now:
            # save model every x epochs
            model.save_model(
                path=os.path.join(
                    log_dir, f"epoch-{e+1}_posenet.model"
                )
            )

        if early_stopping.stop_now:
            print("Early stopping triggered.")
            break

    # log hyperparameter results
    logger.add_hparams(validation_criterion="loss")

    # final evaluation for every available validation sample
    if vary_observation_windows_range is None:
        evaluate_training_run(
            log_dir=log_dir,
            network_type=network_type,
            dataset_type=dataset_type,
            pose_representation=pose_representation,
            history_seconds=history_seconds,
            future_seconds=future_seconds,
            sampling_rate=sampling_rate,
            eval_metrics=[
                metrics.MPJPE(),
                metrics.MAE(pose_linkage=pose_linkage)
            ],
            num_prediction_visualizations=0,
            training_with_occlusions=training_with_occlusions
        )
    else:
        print("Evaluating best Model for different observation windows:")
        for history_length in [
            seconds for seconds in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 1.75, 2.0]
            if seconds >= vary_observation_windows_range[0]
            and seconds <= vary_observation_windows_range[1]
        ]:
            evaluate_training_run(
                log_dir=log_dir,
                network_type=network_type,
                dataset_type=dataset_type,
                pose_representation=pose_representation,
                history_seconds=history_length,
                future_seconds=future_seconds,
                sampling_rate=sampling_rate,
                eval_metrics=[
                    metrics.MPJPE(),
                    metrics.MAE(pose_linkage=pose_linkage)
                ],
                num_prediction_visualizations=0,
                training_with_occlusions=training_with_occlusions
            )


if __name__ == "__main__":
    main(
        network_type=NetworkType.SPATIO_TEMPORAL,
        checkpoint_path=None, #PATH/TO/YOUR/CHECKPOINTS/
        dataset_type=DatasetType.HUMAN36M,
        pose_representation=RepresentationType.CenterJDScaleIndividual,
        loss_functions=[
            (1, torch.nn.L1Loss()),
            # Secondary Loss function
            # (
            #     0.5, CosineLossLimb(
            #         pose_linkage=get_pose_linkage(
            #             dataset_type=DatasetType.HUMAN36M
            #         )
            #     )
            # ),
        ],
        # varying observation window training:
        # vary_observation_windows_range=(0.74, 1.76),
        
        # training with occlusions:
        # training_with_occlusions=True
    )
