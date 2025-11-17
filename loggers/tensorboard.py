import io
from PIL import Image
import plotly.graph_objects as go

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from loggers.logger import Logger


class TensorboardLogger(Logger):

    def __init__(self, log_dir: str, hyperparameters: dict) -> None:
        super().__init__(log_dir=log_dir, hyperparameters=hyperparameters)

        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def _add_scalar(self, name: str, value: float, valid: bool = False) -> None:
        key = f"{name}/step"

        train_or_valid = 'valid' if valid else 'train'
        self.tb_writer.add_scalars(
            key,
            {train_or_valid: value},
            global_step=self.current_step
        )

    def _add_scalar_epoch(self, name: str, value: float, valid: bool = False) -> None:
        key = f"{name}/epoch"

        train_or_valid = 'valid' if valid else 'train'
        self.tb_writer.add_scalars(
            key,
            {train_or_valid: value},
            global_step=self.current_epoch
        )

    def _add_figure(self, fig: go.Figure, fig_name: str) -> None:
        image = Image.open(
            io.BytesIO(fig.to_image(format="png"))
        )

        image_tensor = transforms.ToTensor()(image)

        self.tb_writer.add_image(fig_name, image_tensor, self.current_epoch)

    def _add_hparams(self, hparam_dict: dict, metric_dict: dict, epoch: int) -> None:
        self.tb_writer.add_hparams(
            hparam_dict=hparam_dict,
            metric_dict=metric_dict,
            global_step=epoch
        )
