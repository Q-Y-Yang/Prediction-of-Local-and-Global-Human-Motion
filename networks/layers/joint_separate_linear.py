import torch


class JointSeparateLinear(torch.nn.Module):
    """
    Module that applies separate linear layers at each joint.
    """

    def __init__(
        self,
        n_joints: int,
        in_features: int,
        out_features: int
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.in_features = in_features
        self.out_features = out_features

        # linear layers that are applied at their corresponding joint dimension
        self.joint_embedding_layers = torch.nn.ModuleList([
            torch.nn.Linear(self.in_features, self.out_features)
            for _ in range(n_joints)
        ])

    def forward(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]

        # calculate joint embeddings separately
        sequence_embedded = torch.stack([
            layer(pose_sequence[..., joint_index, :])
            for joint_index, layer in enumerate(self.joint_embedding_layers)
        ], dim=-2)

        return sequence_embedded
