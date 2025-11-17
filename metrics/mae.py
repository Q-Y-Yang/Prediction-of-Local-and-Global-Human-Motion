import torch

from metrics.metrics import Metric

from data import pose_representation, PoseLinkage


class MAE(Metric):
    def __init__(self, pose_linkage: PoseLinkage):
        self.rotation_matrix_representation = pose_representation.RotationMatrices(
            pose_linkage=pose_linkage, relative_orientations=True
        )

    def __str__(self) -> str:
        return "MAE"

    def _rotation_matrices_to_euler(
        self,
        rotation_matrices: torch.Tensor
    ) -> torch.Tensor:
        original_shape = rotation_matrices.shape
        # reshape rotation matrices to [BATCH, 3, 3]
        rotation_matrix = rotation_matrices.view(-1, 3, 3)

        # extract the individual elements of the rotation matrices for readability
        R11 = rotation_matrix[:, 0, 0]
        R12 = rotation_matrix[:, 0, 1]
        R13 = rotation_matrix[:, 0, 2]
        R21 = rotation_matrix[:, 1, 0]
        R31 = rotation_matrix[:, 2, 0]
        R32 = rotation_matrix[:, 2, 1]
        R33 = rotation_matrix[:, 2, 2]

        # compute initial vales for alpha, beta and gamma (yaw, pitch and roll)
        beta = - torch.asin(R31)

        cos_beta = torch.cos(beta)
        alpha = torch.atan2(R21 / cos_beta, R11 / cos_beta)
        gamma = torch.atan2(R32 / cos_beta, R33 / cos_beta)

        # check for gimbal lock conditions
        # 0.99999 approximates 1 in case of numerical errors
        gimbal_lock_positive = R31 >= 0.99999
        gimbal_lock_negative = R31 <= -0.99999

        if torch.any(gimbal_lock_positive):
            gimbal_idx = torch.where(gimbal_lock_positive)
            beta[gimbal_idx] = - torch.pi / 2
            alpha[gimbal_idx] = 0
            gamma[gimbal_idx] = torch.atan2(-R12[gimbal_idx], -R13[gimbal_idx])

        if torch.any(gimbal_lock_negative):
            gimbal_idx = torch.where(gimbal_lock_negative)
            beta[gimbal_idx] = torch.pi / 2
            alpha[gimbal_idx] = 0
            gamma[gimbal_idx] = torch.atan2(R12[gimbal_idx], R13[gimbal_idx])

        euler_angles = torch.stack([alpha, beta, gamma], dim=1)
        # reshape tensor to original input shape (except for the feature dim)
        euler_angles = euler_angles.view(*original_shape[:-1], 3)

        return euler_angles

    def _calculate_metric(
        self,
        prediction_3d: torch.Tensor,
        groundtruth_3d: torch.Tensor
    ) -> torch.Tensor:
        prediction_rot_mat, _ = (
            self.rotation_matrix_representation.from_3d_coordinates_sequence(prediction_3d)
        )
        groundtruth_rot_mat, _ = (
            self.rotation_matrix_representation.from_3d_coordinates_sequence(groundtruth_3d)
        )
        
        prediction_euler = self._rotation_matrices_to_euler(prediction_rot_mat)
        groundtruth_euler = self._rotation_matrices_to_euler(groundtruth_rot_mat)

        return torch.nn.functional.mse_loss(
            prediction_euler,
            groundtruth_euler,
            reduction="none"
        ).mean(dim=-1)
