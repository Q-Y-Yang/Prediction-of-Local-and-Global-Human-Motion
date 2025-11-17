import torch

from networks.layers import PositionalEncoding


class BaselineTransformer(torch.nn.Module):

    def __init__(
        self,
        n_joints: int,
        joint_feature_dim: int,
        embedding_dim: int,
        max_length: int,
        n_transformer_blocks: int = 6,
        n_attention_heads: int = 8,
        transformer_feedforward_dim: int = 2048,
        output_activation: str = "linear",
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_joints = n_joints
        self.joint_feature_dim = joint_feature_dim
        self.input_dim = self.n_joints * self.joint_feature_dim
        self.d_model = embedding_dim

        self.n_transformer_blocks = n_transformer_blocks
        self.n_attention_heads = n_attention_heads
        self.transformer_feedforward_dim = transformer_feedforward_dim

        self.output_activation = output_activation

        self.max_length = max_length

        self.dropout = dropout

        # embedding layer, mapping input dimension to d_model dimension
        self.embedding_linear = torch.nn.Linear(self.input_dim, self.d_model)

        self.positional_encoding = PositionalEncoding(
            encoding_size=self.d_model,
            dropout=self.dropout,
            batch_first=True,
            max_len=self.max_length
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_attention_heads,
            dim_feedforward=transformer_feedforward_dim,
            batch_first=True,
            dropout=self.dropout
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_blocks,
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_attention_heads,
            dim_feedforward=transformer_feedforward_dim,
            batch_first=True,
            dropout=self.dropout
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=n_transformer_blocks,
        )

        # linear layer to get back to input dimension shape
        self.linear_to_input_dim = torch.nn.Linear(
            self.d_model, self.input_dim
        )

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(
        self,
        history: torch.Tensor,
        noisy: torch.Tensor
    ) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        B, N, _, _ = history.shape
        B, T, _, _ = noisy.shape

        # [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        # -> [BATCH, TIMESTEP, JOINT_IDX * FEATURES]
        history = history.view(B, N, self.input_dim)
        noisy = noisy.view(B, T, self.input_dim)

        # embed inputs to get from input dimension to d_model dimension
        # [BATCH, TIMESTEP, JOINT_IDX * FEATURES]
        # -> [BATCH, TIMESTEP, JOINT_IDX * EMBEDDING_DIM]
        history_embedded = self.embedding_linear(history)
        noisy_embedded = self.embedding_linear(noisy)

        # add positional encoding
        enc_inputs = self.positional_encoding(history_embedded)
        dec_inputs = self.positional_encoding(noisy_embedded)

        # encode pose history
        enc_output = self.encoder(enc_inputs)

        # decode future predictions
        dec_output = self.decoder(dec_inputs, enc_output)

        # backprojection to input dimension
        # [BATCH, TIMESTEP, JOINT_IDX * EMBEDDING_DIM]
        # -> [BATCH, TIMESTEP, JOINT_IDX * FEATURES]
        output_sequence = self.linear_to_input_dim(dec_output)

        # residual connection
        output_sequence = output_sequence + noisy

        match self.output_activation.lower():
            case "tanh":
                output_sequence = torch.nn.functional.tanh(output_sequence)

        # [BATCH, TIMESTEP, JOINT_IDX * FEATURES]
        # -> [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        output_sequence = output_sequence.view(
            B, T, self.n_joints, self.joint_feature_dim
        )

        return output_sequence

    def get_configuration(self) -> dict:
        return {
            "n_joints": self.n_joints,
            "joint_feature_dim": self.joint_feature_dim,
            "embedding_dim": self.d_model,
            "max_length": self.max_length,
            "n_transformer_blocks": self.n_transformer_blocks,
            "n_attention_heads": self.n_attention_heads,
            "transformer_feedforward_dim": self.transformer_feedforward_dim,
            "output_activation": self.output_activation,
            "dropout": self.dropout
        }

    def save_model(self, path: str):
        data = {
            'state_dict': self.state_dict(),
            "configuration": self.get_configuration()
        }
        torch.save(data, path)

    @classmethod
    def load_model(cls, path: str):
        checkpoint = torch.load(path, weights_only=False)

        model = cls(**checkpoint["configuration"])

        model.load_state_dict(checkpoint['state_dict'])

        return model
