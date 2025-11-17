from typing import Optional

import torch

from data import PoseLinkage

# from: https://github.com/airalcorn2/pytorch-geodesic-loss


class GeodesicLoss(torch.nn.Module):
    """
    Creates a criterion that measures the distance between rotation matrices.
    The distance ranges from 0 to pi.

    References:
    http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices

    "Metrics for 3D Rotations: Comparison and Analysis" 
    (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices of size [BATCH, 3, 3].

    The loss can be described as:

    loss = arccos ( (trace(R_prediction R_target.T) - 1) / 2 )

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            "none" | "mean" | "sum"

    Shape:
        - Input:  Shape [N, 3, 3]
        - Target: Shape [N, 3, 3]
        - Output: If reduction is none then N. Otherwise, scalar.
    """

    def __init__(
        self,
        pose_linkage: PoseLinkage,
        eps: float = 1e-7,
        reduction: Optional[str] = "mean"
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

        self.root_index = pose_linkage.root_index

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, _, J, _ = prediction.shape

        # filter out the root joint of the pose linkage
        # it should not contribute to the loss function
        mask = torch.ones(J, dtype=torch.bool)
        mask[self.root_index] = False

        p = prediction[:, :, mask, :].reshape(-1, 3, 3)
        t = target[:, :, mask, :].reshape(-1, 3, 3)

        R_diffs = p @ t.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(
            torch.clamp(
                (traces - 1) / 2, -1 + self.eps, 1 - self.eps
            )
        )

        match self.reduction:
            case "none":
                return dists
            case "mean":
                return dists.mean()
            case "sum":
                return dists.sum()
