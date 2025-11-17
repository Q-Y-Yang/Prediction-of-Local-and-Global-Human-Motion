from typing import Optional

import torch

from data import PoseLinkage

# adapted from:
# https://github.com/abduallahmohamed/Skeleton-Graph/blob/c6f37ec59ca9db9625f538750d690b2d1f24f27c/train.py#L160


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(x, y, dim=-1)


class CosineLossLimb(torch.nn.Module):
    def __init__(
        self,
        pose_linkage: PoseLinkage
    ) -> None:
        super().__init__()
        self.pose_linkage = pose_linkage

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        links = self.pose_linkage.links

        cos_loss = 0
        for i, k in links:
            cos_loss -= cosine_similarity(
                target[:, :, k, :] - target[:, :, i, :],
                prediction[:, :, k, :] - prediction[:, :, i, :],
            )

        cos_loss = (1/len(links)) * cos_loss

        # add 1 to have value range [0, 2] instead of [-1, 1]
        return cos_loss.mean() + 1


class CosineLossOriginal(torch.nn.Module):
    def __init__(
        self,
        pose_linkage: Optional[PoseLinkage] = None
    ) -> None:
        self.pose_linkage = pose_linkage

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pose_linkage is not None:
            links = self.pose_linkage.links
        else:
            # use original joint pairing (i, i + 1) if no pose linkage is given
            n_joints = prediction.shape[2]
            links = [(i, (i + 1) % n_joints) for i in range(n_joints)]

        cos_loss = 0
        for i, k in links:
            cos_loss += torch.abs(
                cosine_similarity(target[:, :, i, :], target[:, :, k, :]) -
                cosine_similarity(
                    prediction[:, :, i, :], prediction[:, :, k, :]
                )
            )

        cos_loss = cos_loss.sqrt()
        cos_loss = (1/len(links)) * cos_loss

        return cos_loss.mean()
