import torch

from networks.layers import JointSeparateLinear


class SpatialAttention(torch.nn.Module):
    def __init__(
        self,
        n_joints: int,
        joint_embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        q_shared: bool = False,
        k_shared: bool = True,
        v_shared: bool = True
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.joint_embedding_dim = joint_embedding_dim

        self.num_heads = num_heads

        if q_shared:
            self.q_layer = torch.nn.Linear(self.joint_embedding_dim,
                                           self.joint_embedding_dim)
        else:
            self.q_layer = JointSeparateLinear(
                n_joints=self.n_joints,
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )

        if k_shared:
            self.k_layer = torch.nn.Linear(self.joint_embedding_dim,
                                           self.joint_embedding_dim)
        else:
            self.k_layer = JointSeparateLinear(
                n_joints=self.n_joints,
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )

        if v_shared:
            self.v_layer = torch.nn.Linear(self.joint_embedding_dim,
                                           self.joint_embedding_dim)
        else:
            self.v_layer = JointSeparateLinear(
                n_joints=self.n_joints,
                in_features=self.joint_embedding_dim,
                out_features=self.joint_embedding_dim
            )

        self.multi_head_attentions = torch.nn.MultiheadAttention(
            self.joint_embedding_dim, num_heads=self.num_heads, dropout=dropout,
            batch_first=True
        )

        self.dropout = torch.nn.Dropout(p=dropout)

        self.layer_norm = torch.nn.LayerNorm(self.joint_embedding_dim)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURE]

        B, _, _, _ = input_sequence.shape

        residual = input_sequence

        # [BATCH, TIMESTEP, JOINT_IDX, FEATURE]
        # -> [BATCH * TIMESTEP, JOINT_IDX, FEATURE]
        poses = input_sequence.reshape(
            -1, self.n_joints, self.joint_embedding_dim
        )

        # [BATCH, TIMESTEP, JOINT_IDX, FEATURE]
        # -> [BATCH * JOINT_IDX, TIMESTEPS, FEATURE]
        q = self.q_layer(poses).view(
            B * self.n_joints, -1, self.joint_embedding_dim
        )
        k = self.k_layer(poses).view(
            B * self.n_joints, -1, self.joint_embedding_dim
        )
        v = self.v_layer(poses).view(
            B * self.n_joints, -1, self.joint_embedding_dim
        )

        attention_output, _ = self.multi_head_attentions(
            q, k, v, need_weights=False
        )

        # [BATCH * JOINT_IDX, TIMESTEPS, FEATURE]
        # -> [BATCH, TIMESTEP, JOINT_IDX, FEATURE]
        attention_output = attention_output.reshape(*input_sequence.shape)

        attention_output = self.dropout(attention_output)

        attention_output = self.layer_norm(attention_output + residual)

        return attention_output
