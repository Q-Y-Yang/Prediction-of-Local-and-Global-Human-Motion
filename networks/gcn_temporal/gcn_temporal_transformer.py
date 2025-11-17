import torch

from networks.layers import PoseEncoderGCN, PoseDecoder, PositionalEncoding

from networks.gcn_temporal.encoder import GCNTemporalEncoderBlock
from networks.gcn_temporal.decoder import GCNTemporalDecoderBlock


class GCNTemporalTransformer(torch.nn.Module):
    """
    A non-autoregressive version of the Spatio-Temporal Transformer proposed by
    Aksan et al. in
    "A Spatio-temporal Transformer for 3D Human Motion Prediction" (2021)
    The Spatial Attention modules are replaced by Graph Convolution layers
    """

    def __init__(
        self,
        n_joints: int,
        joint_feature_dim: int,
        joint_embedding_dim: int,
        max_length: int,
        n_attention_blocks: int = 6,
        num_heads: int = 8,
        output_activation="linear",
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.n_joints = n_joints
        self.joint_feature_dim = joint_feature_dim
        self.joint_embedding_dim = joint_embedding_dim

        self.n_attention_blocks = n_attention_blocks
        self.output_activation = output_activation
        self.num_heads = num_heads

        self.max_length = max_length

        self.dropout_p = dropout

        self.pose_encoder = PoseEncoderGCN(
            in_features=self.joint_feature_dim,
            out_features=self.joint_embedding_dim,
            n_joints=self.n_joints,
            dropout=self.dropout_p
        )

        self.positional_encoding = PositionalEncoding(
            encoding_size=self.joint_embedding_dim,
            joint_seperate=True,
            batch_first=True,
            dropout=dropout,
            max_len=self.max_length
        )

        self.enc_blocks = torch.nn.ModuleList([
            GCNTemporalEncoderBlock(
                n_joints=self.n_joints,
                joint_embedding_dim=self.joint_embedding_dim,
                num_heads=self.num_heads,
                dropout=dropout
            )
            for _ in range(n_attention_blocks)
        ])

        self.dec_blocks = torch.nn.ModuleList([
            GCNTemporalDecoderBlock(
                n_joints=self.n_joints,
                joint_embedding_dim=self.joint_embedding_dim,
                num_heads=self.num_heads,
                dropout=dropout
            )
            for _ in range(n_attention_blocks)
        ])

        self.pose_decoder = PoseDecoder(
            in_features=self.joint_embedding_dim,
            out_features=self.joint_feature_dim,
            n_joints=self.n_joints
        )

    def encoder(
        self,
        input_sequence: torch.Tensor
    ) -> torch.Tensor:
        encoder_block_output = input_sequence
        for i in range(self.n_attention_blocks):
            encoder_block = self.enc_blocks[i]
            encoder_block_output = encoder_block(
                encoder_block_output
            )

        return encoder_block_output

    def decoder(
        self,
        decoder_input: torch.Tensor,
        encoder_output: torch.Tensor
    ) -> torch.Tensor:
        decoder_block_output = decoder_input
        for i in range(self.n_attention_blocks):
            decoder_block = self.dec_blocks[i]
            decoder_block_output = decoder_block(
                decoder_block_output, encoder_output
            )

        return decoder_block_output

    def forward(
        self,
        history: torch.Tensor,
        noisy: torch.Tensor
    ) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        
        # [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        # -> [BATCH, TIMESTEP, JOINT_IDX, EMBEDDING_DIM]

        history_embedded = self.pose_encoder(history)
        noisy_embedded = self.pose_encoder(noisy)

        # apply temporal positional encoding
        enc_inputs = self.positional_encoding(history_embedded)
        dec_inputs = self.positional_encoding(noisy_embedded)

        enc_output = self.encoder(enc_inputs)

        dec_output = self.decoder(dec_inputs, enc_output)

        # project joint embeddings back to original feature space
        # [BATCH, TIMESTEP, JOINT_IDX, EMBEDDING_DIM]
        # -> [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        output_sequence = self.pose_decoder(dec_output)

        # residual connection
        output_sequence = output_sequence + noisy
        
        if self.output_activation == "tanh":
            output_sequence = torch.tanh(output_sequence)

        return output_sequence

    def get_configuration(self) -> dict:
        return {
            "n_joints": self.n_joints,
            "joint_feature_dim": self.joint_feature_dim,
            "joint_embedding_dim": self.joint_embedding_dim,
            "max_length": self.max_length,
            "n_attention_blocks": self.n_attention_blocks,
            "num_heads": self.num_heads,
            "output_activation": self.output_activation,
            "dropout": self.dropout_p
        }

    def save_model(self, path: str):
        data = {
            'state_dict': self.state_dict(),
            "configuration": self.get_configuration()
        }
        torch.save(data, path)

    @classmethod
    def load_model(cls, path: str):
        checkpoint = torch.load(path, weights_only=True)

        model = cls(**checkpoint["configuration"])

        model.load_state_dict(checkpoint['state_dict'])

        return model
