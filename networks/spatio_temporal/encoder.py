import torch

from networks.layers import SpatialAttention, TemporalAttention, PositionwiseFeedforward


class SpatioTemporalEncoderBlock(torch.nn.Module):
    """
    Encoder Architecture

    proposed by 
    Aksan et al. 
    in "A Spatio-temporal Transformer for 3D Human Motion Prediction" (2021)
    """

    def __init__(
        self,
        n_joints: int,
        joint_embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.joint_embedding_dim = joint_embedding_dim
        
        self.num_heads = num_heads

        self.spatial_attention = SpatialAttention(
            n_joints=self.n_joints,
            joint_embedding_dim=self.joint_embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout
        )

        self.temporal_attention = TemporalAttention(
            n_joints=self.n_joints,
            joint_embedding_dim=self.joint_embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout
        )

        self.feed_forward = PositionwiseFeedforward(
            n_joints=self.n_joints,
            joint_embedding_dim=self.joint_embedding_dim,
            joint_weights_shared=False,
            dropout=dropout
        )

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        spatial_attention = self.spatial_attention(input_sequence)
        temporal_attention = self.temporal_attention(
            input_sequence, input_sequence, input_sequence
        )

        # add both attention results together
        added = spatial_attention + temporal_attention

        output = self.feed_forward(added)

        return output
