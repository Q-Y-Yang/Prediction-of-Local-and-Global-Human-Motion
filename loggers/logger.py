import os
import plotly.graph_objects as go

import torch

from collections import defaultdict
from abc import ABC, abstractmethod


class Logger(ABC):

    scalars_by_epoch = defaultdict(list)

    def __init__(self, log_dir: str, hyperparameters: dict) -> None:
        self.log_dir = log_dir
        self.hyperparameters = hyperparameters

        self.current_step = 0
        self.current_epoch = 1
        self.scalars_by_epoch = defaultdict(list)

    def step(self):
        self.current_step += 1

    def step_epoch(self):
        self.current_epoch += 1

    @abstractmethod
    def _add_scalar(self, name: str, value: float, valid: bool = False) -> None:
        pass

    def add_scalar(self, name: str, value: float, valid: bool = False) -> None:
        self._add_scalar(name=name, value=value, valid=valid)

    @abstractmethod
    def _add_scalar_epoch(self, name: str, value: float, valid: bool = False) -> None:
        pass

    def add_scalar_epoch(
        self,
        name: str,
        values: list[float],
        n_samples: int,
        valid: bool = False
    ) -> torch.Tensor:
        mean = torch.sum(torch.tensor(values)) / n_samples

        train_or_valid = 'valid' if valid else 'train'
        self.scalars_by_epoch[f"{name}/{train_or_valid}"].append(
            (self.current_epoch, mean)
        )

        self._add_scalar_epoch(name=name, value=mean, valid=valid)

        return mean

    @abstractmethod
    def _add_figure(self, fig: go.Figure, fig_name: str) -> None:
        pass

    def add_figure(self, fig: go.Figure, fig_name: str) -> None:
        self._add_figure(fig=fig, fig_name=fig_name)

        filepath = os.path.join(self.log_dir, "plots")
        os.makedirs(filepath, exist_ok=True)

        fig.write_image(
            os.path.join(filepath, f"{fig_name}-{self.current_epoch}.png")
        )

    @abstractmethod
    def _add_hparams(self, hparam_dict: dict, metric_dict: dict, epoch: int) -> None:
        pass

    def add_hparams(self, validation_criterion: str):
        criterion = f"{validation_criterion}/valid"

        epoch = min(
            self.scalars_by_epoch[criterion],
            key=lambda epoch_and_value: epoch_and_value[1]
        )[0]

        hparam_dict = {
            **self.hyperparameters,
            "epoch": epoch,
        }

        # collect all metric values from best validation epoch
        metric_dict = {}
        metrics = self.scalars_by_epoch.keys()
        for metric in metrics:
            for e, value in self.scalars_by_epoch[metric]:
                if e == epoch:
                    metric_dict[f"hparam/{metric}"] = value
                    break

        self._add_hparams(
            hparam_dict=hparam_dict,
            metric_dict=metric_dict,
            epoch=epoch
        )


class MultiLogger:

    def __init__(self, log_dir: str, loggers: list[Logger]) -> None:
        super().__init__()

        self.log_dir = log_dir
        self.loggers = loggers

        self.current_step = 0
        self.current_epoch = 1

    def step(self) -> None:
        self.current_step += 1
        for logger in self.loggers:
            logger.step()

    def step_epoch(self) -> None:
        self.current_epoch += 1
        for logger in self.loggers:
            logger.step_epoch()

    def add_scalar(self, name: str, value: float, valid: bool = False) -> None:
        for logger in self.loggers:
            logger.add_scalar(name=name, value=value, valid=valid)

    def add_scalar_epoch(
        self,
        name: str,
        values: list[float],
        n_samples: int,
        valid: bool = False
    ) -> torch.Tensor:
        mean = -1
        for logger in self.loggers:
            mean = logger.add_scalar_epoch(
                name=name,
                values=values,
                n_samples=n_samples,
                valid=valid
            )

        return mean

    def add_figure(self, fig, fig_name: str) -> None:
        for logger in self.loggers:
            logger.add_figure(fig=fig, fig_name=fig_name)

    def add_hparams(self, validation_criterion: str) -> None:
        for logger in self.loggers:
            logger.add_hparams(validation_criterion=validation_criterion)
