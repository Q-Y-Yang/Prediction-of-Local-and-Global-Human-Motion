import torch

from networks.layers import JointSeparateLinear


class PositionwiseFeedforward(torch.nn.Module):
    """
    Implements a Feedforward layer similar to the one proposed in
    "Attention is all you need" (2017) by Vaswani et al.

    changes:
    the linear layers can be joint separate

    the hidden dimension is equal to the model dimension opposed to the original version
    where hidden dimension = 2048 and model dimension = 512
    """

    def __init__(
        self,
        n_joints: int,
        joint_embedding_dim: int,
        joint_weights_shared: bool = False,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.joint_embedding_dim = joint_embedding_dim

        if joint_weights_shared:
            self.feed_forward_1 = torch.nn.Linear(
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )
            self.feed_forward_2 = torch.nn.Linear(
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )
        else:
            self.feed_forward_1 = JointSeparateLinear(
                n_joints=self.n_joints,
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )
            self.feed_forward_2 = JointSeparateLinear(
                n_joints=self.n_joints,
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )

        self.layer_norm = torch.nn.LayerNorm(self.joint_embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]

        residual = x

        processed = self.feed_forward_1(x)
        processed = torch.nn.functional.relu(processed)
        processed = self.feed_forward_2(processed)

        processed = self.dropout(processed)
        processed += residual

        output = self.layer_norm(processed)

        return output
