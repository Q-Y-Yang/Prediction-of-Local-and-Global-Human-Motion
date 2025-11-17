import torch

from metrics.metrics import Metric

class MPJPE(Metric):
    def __str__(self) -> str:
        return "MPJPE"
    
    def _calculate_metric(
        self,
        prediction_3d: torch.Tensor,     # [B, T, J, 3]
        groundtruth_3d: torch.Tensor     # [B, T, J, 3]
    ) -> torch.Tensor:
        # Compute Euclidean distance for each joint
        error_per_joint = torch.norm(prediction_3d - groundtruth_3d, dim=-1)  # [B, T, J]

        # Mean over joints
        error_per_frame = error_per_joint.mean(dim=-1)  # [B, T]

        return error_per_frame  # one MPJPE value per frame, per sequence

