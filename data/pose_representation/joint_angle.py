import torch

from data.data import Frame, MotionSequence, PoseLinkage
from data.pose_representation.base import PoseRepresentation


class RotationMatrices(PoseRepresentation):

    pose_linkage: PoseLinkage

    def __init__(
        self,
        pose_linkage: PoseLinkage,
        relative_orientations: bool = True
    ) -> None:
        self.pose_linkage = pose_linkage
        self.relative_orientations = relative_orientations

    @property
    def joint_feature_dim(self) -> int:
        return 9

    @staticmethod
    def calculate_rotation_matrix(
        reference_vector: torch.Tensor,
        direction_vector: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = reference_vector.shape

        # calculations are derived from Rodrigues' rotation formula
        # and adapted for two arbitrary vectors.
        # the function computes the rotation matrix which is able to rotate
        # the reference vector so that it aligns with the direction vector.
        # formulas adapted from: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

        reference_vec = reference_vector.view(-1, 3)
        direction_vec = direction_vector.view(-1, 3)

        # norm the vectors to unit length
        a = (
            reference_vec /
            torch.linalg.norm(reference_vec, dim=-1, keepdim=True)
        )
        b = (
            direction_vec /
            torch.linalg.norm(direction_vec, dim=-1, keepdim=True)
        )

        # identity matrix
        batch_size = reference_vec.shape[0]
        identity = torch.eye(
            3, device=reference_vec.device
        ).expand(batch_size, 3, 3)

        # dot product
        dot = torch.sum(a * b, dim=-1)
        # cross product
        cross = torch.cross(a, b, dim=-1)
        norm_cross = torch.norm(cross, dim=-1)
        # outer product for cross product vector
        outer_cross = cross.unsqueeze(2) * cross.unsqueeze(1)

        # create skew-symmetric matrix
        zero = torch.zeros(batch_size, 1, device=reference_vector.device)

        c0 = cross[:, 0].unsqueeze(1)
        c1 = cross[:, 1].unsqueeze(1)
        c2 = cross[:, 2].unsqueeze(1)

        row_1 = torch.cat([zero, -c2, c1], dim=1)
        row_2 = torch.cat([c2, zero, -c0], dim=1)
        row_3 = torch.cat([-c1, c0, zero], dim=1)

        skew = torch.stack([row_1, row_2, row_3], dim=1)

        # final computation to obtain rotation matrix
        term_1 = dot.unsqueeze(1).unsqueeze(1) * identity
        term_2 = skew
        term_3 = (
            (1 - dot) / (norm_cross ** 2)
        ).unsqueeze(1).unsqueeze(1) * outer_cross

        R = term_1 + term_2 + term_3

        R = R.view(*original_shape[:-1], 3, 3)

        return R

    def from_3d_coordinates_sequence(
        self,
        pos_3d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # create a new tensor with matching shape but feature dim of 9
        pos_rotation_matrices = torch.zeros(*pos_3d.shape[:-1], 9)

        n_joints = self.pose_linkage.n_joints

        # backtransformation data will store the pose center position
        # + distances for each joint to their parent joint
        # size = n_joints + 3
        backtransformation_data = torch.zeros(
            *pos_3d.shape[:-2], n_joints + 3
        )

        for link in self.pose_linkage.links:
            a, b = link

            if not self.relative_orientations or a == self.pose_linkage.root_index:
                # for calculating absolute directions compare with [0, 0, 1] reference
                # for relative directions, center index also has [0, 0, 1] reference
                reference_vector = torch.tensor(
                    [0.0, 0.0, 1.0], device=pos_3d.device
                )
                reference_vector = reference_vector.repeat(
                    *pos_3d.shape[:-2], 1
                )
            else:
                # (not for absolute directions)
                # every other joint has the preceding direction vector
                # as reference vector
                parent_of_a = self.pose_linkage.parent_of(a)
                reference_vector = (
                    pos_3d[..., a, :] -
                    pos_3d[..., parent_of_a, :]
                )

            direction_vector = pos_3d[..., b, :] - pos_3d[..., a, :]

            pos_rotation_matrices[..., b, :] = self.calculate_rotation_matrix(
                reference_vector=reference_vector,
                direction_vector=direction_vector
            ).view(*reference_vector.shape[:-1], 9)

            # distance between joints a and b
            backtransformation_data[..., b] = torch.linalg.norm(
                direction_vector, dim=-1
            )

        # center of the pose
        backtransformation_data[..., -3:] = pos_3d[
            ..., self.pose_linkage.root_index, :
        ]

        return pos_rotation_matrices, backtransformation_data

    def preprocess_sequence(self, sequence: MotionSequence) -> None:
        sequence_3d = torch.stack(
            [frame.pose for frame in sequence.frames], dim=0
        )

        sequence_rot, backtransform_data = (
            self.from_3d_coordinates_sequence(sequence_3d)
        )

        for frame, pose_rot, backtransform in zip(
            sequence.frames, sequence_rot, backtransform_data
        ):
            frame.pose = pose_rot
            frame.backtransformation_data = backtransform

    def from_3d_coordinates(
        self,
        history: list[Frame],
        groundtruth: list[Frame]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_rot = torch.stack(
            [frame.pose for frame in history], dim=0
        )
        groundtruth_rot = torch.stack(
            [frame.pose for frame in groundtruth], dim=0
        )
        noisy_rot = history_rot[-1].expand_as(groundtruth_rot).clone()

        history_backtransform = torch.stack(
            [frame.backtransformation_data for frame in history], dim=0
        )
        future_backtransform = torch.stack(
            [frame.backtransformation_data for frame in groundtruth], dim=0
        )

        return (
            history_rot,
            noisy_rot,
            groundtruth_rot,
            history_backtransform,
            future_backtransform
        )

    def to_3d_coordinates(
        self,
        pos_rotation_matrices: torch.Tensor,
        backtransformation_data: torch.Tensor
    ) -> torch.Tensor:
        # create a new tensor for 3d joint coordinates
        # the tensor has the same shape but with joint feature dim of 3
        pos_3d = torch.zeros(
            *pos_rotation_matrices.shape[:-1], 3,
            device=pos_rotation_matrices.device
        )

        # initalize tensor that stores reference vectors
        reference_vectors = torch.zeros(
            *pos_rotation_matrices.shape[:-1], 3,
            device=pos_rotation_matrices.device
        )

        if self.relative_orientations:
            reference_vectors[..., self.pose_linkage.root_index, :] = torch.tensor(
                [0.0, 0.0, 1.0]
            )
        else:
            reference_vectors[..., :, :] = torch.tensor(
                [0.0, 0.0, 1.0]
            )

        center_position = backtransformation_data[..., -3:]

        # compute the 3D joint coordinates succesively
        for joint_index in self.pose_linkage.joints_in_topological_order:
            if joint_index == self.pose_linkage.root_index:
                pos_3d[..., joint_index, :] = center_position
                continue

            parent_index = self.pose_linkage.parent_of(joint_index)
            reference_vector = reference_vectors[..., parent_index, :]

            # reshape flat rotation matrix from [..., 9] to [..., 3, 3]
            R = pos_rotation_matrices[..., joint_index, :].reshape(
                *pos_rotation_matrices.shape[:-2], 3, 3
            )

            # joint distance to parent joint is stored in backtransformation data
            distance = backtransformation_data[..., joint_index]

            # apply rotation to reference vector
            # .view() operation is needed as batch matrix multiplication only works
            # with one batch dimension
            # dimensions could be [BATCH_SIZE, TIMESTEP, JOINT_INDEX, ROTATION_MATRIX]
            # then dimensions need to be merged to [BATCH_SIZE * TIMESTEP, ...]
            # so that torch.bmm can be used
            direction = torch.bmm(
                R.view(-1, 3, 3),
                reference_vector.view(-1, 3, 1)
            ).squeeze(-1)

            # reshape in case batch dimensions were merged above
            # [BATCH_SIZE * TIMESTEP, ...] -> [BATCH_SIZE, TIMESTEP, ...]
            direction = direction.reshape(reference_vector.shape)

            if self.relative_orientations:
                # store joint direction as reference vector for child joints
                reference_vectors[..., joint_index, :] = direction

            # compute joint 3d position

            # intialize with parent position
            pos_3d[..., joint_index, :] = pos_3d[..., parent_index, :]
            # add direction vector * joint distance to parent joint
            pos_3d[..., joint_index, :] += direction * distance.unsqueeze(-1)

        return pos_3d
