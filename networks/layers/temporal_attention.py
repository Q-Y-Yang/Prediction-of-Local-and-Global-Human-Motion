import torch

from networks.layers import JointSeparateLinear


class TemporalAttention(torch.nn.Module):
    def __init__(
        self,
        n_joints: int,
        joint_embedding_dim: int,
        q_shared: bool = False,
        k_shared: bool = False,
        v_shared: bool = False,
        attention_shared: bool = False,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.joint_embedding_dim = joint_embedding_dim

        self.num_heads = num_heads
        
        self.attention_shared = attention_shared

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
        
        if attention_shared:
            self.multi_head_attention = torch.nn.MultiheadAttention(
                self.joint_embedding_dim, num_heads=self.num_heads, dropout=dropout,
                batch_first=True
            )
        else:
            self.joint_multi_head_attentions = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(
                    self.joint_embedding_dim, num_heads=self.num_heads, dropout=dropout,
                    batch_first=True
                )
                for _ in range(self.n_joints)
            ])

        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(self.joint_embedding_dim)

    def forward(
        self,
        q_sequence: torch.Tensor,
        k_sequence: torch.Tensor,
        v_sequence: torch.Tensor,
    ) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        
        B, _, _, _ = q_sequence.shape

        residual = q_sequence

        q = self.q_layer(q_sequence)
        k = self.k_layer(k_sequence)
        v = self.v_layer(v_sequence)

        if self.attention_shared:
            # [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
            # -> [BATCH * JOINT_IDX, TIMESTEP, EMBEDDING]
            attention_output = self.multi_head_attention(
                q.view(B * self.n_joints, -1, self.joint_embedding_dim),
                k.view(B * self.n_joints, -1, self.joint_embedding_dim),
                v.view(B * self.n_joints, -1, self.joint_embedding_dim),
                need_weights=False
            )[0].reshape(*q_sequence.shape) # -> [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        else:
            attention_output = torch.stack([
                self.joint_multi_head_attentions[joint_index](
                    q[..., joint_index, :],
                    k[..., joint_index, :],
                    v[..., joint_index, :],
                    need_weights=False
                )[0]
                for joint_index in range(self.n_joints)
            ], dim=-2)

        attention_output = self.dropout(attention_output)

        attention_output = self.layer_norm(attention_output + residual)

        return attention_output
