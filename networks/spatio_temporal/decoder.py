import torch

from networks.layers import SpatialAttention, TemporalAttention, PositionwiseFeedforward


class SpatioTemporalDecoderBlock(torch.nn.Module):
    """
    Decoder Architecture

    inspired by Encoder Architecture of Aksan et al. 
    in "A Spatio-temporal Transformer for 3D Human Motion Prediction" (2021)

    and Decoder Architecture by Vaswani et al.
    in "Attention is all you need" (2017)
    """

    def __init__(
        self,
        n_joints: int,
        joint_embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.joint_embedding_dim = joint_embedding_dim

        self.num_heads = num_heads

        self.spatial_self_attention = SpatialAttention(
            n_joints=self.n_joints,
            joint_embedding_dim=self.joint_embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout
        )

        self.temporal_self_attention = TemporalAttention(
            n_joints=self.n_joints,
            joint_embedding_dim=self.joint_embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout
        )

        self.temporal_enc_attention = TemporalAttention(
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

    def forward(
        self,
        input_sequence: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        # self attention
        spatial_self_attention = self.spatial_self_attention(
            input_sequence
        )

        temporal_self_attention = self.temporal_self_attention(
            input_sequence, input_sequence, input_sequence
        )

        # add both attention results together
        self_attention = spatial_self_attention + temporal_self_attention

        # encoder-decoder temporal attention
        temporal_enc_attention = self.temporal_enc_attention(
            q_sequence=self_attention,
            k_sequence=encoder_output,
            v_sequence=encoder_output
        )

        output = self.feed_forward(temporal_enc_attention)

        return output
