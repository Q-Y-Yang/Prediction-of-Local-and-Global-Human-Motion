import math
import torch


class PositionalEncoding(torch.nn.Module):
    # adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model

    def __init__(
        self,
        encoding_size: int,
        joint_seperate: bool = False,
        batch_first: bool = False,
        dropout: float = 0.1,
        max_len: int = 300
    ):
        super().__init__()

        self.joint_seperate = joint_seperate
        self.batch_first = batch_first

        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # max_len is used to scale the frequencies based on expected sequence lengths
        div_term = torch.exp(
            torch.arange(0, encoding_size, 2) *
            (-math.log(max_len / 2) / encoding_size)
        )

        # use even d_model and slice of last dimension later
        if encoding_size % 2 != 0:
            pe = torch.zeros(max_len, 1, encoding_size + 1)
        else:
            pe = torch.zeros(max_len, 1, encoding_size)

        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        if encoding_size % 2 != 0:
            pe = pe[:, :, :-1]

        if self.joint_seperate:
            pe = pe.unsqueeze(-2)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            if self.joint_seperate:
                # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
                # -> # [TIMESTEP, BATCH, JOINT_IDX, FEATURES]
                permutation = (1, 0, 2, 3)
            else:
                # input shape: [BATCH, TIMESTEP, JOINT_IDX * FEATURES]
                # -> # [BATCH, TIMESTEP, JOINT_IDX * FEATURES]
                permutation = (1, 0, 2)

            # permute to sequence first order
            x = x.permute(*permutation)

        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)

        if self.batch_first:
            # permute back to batch input ordering
            x = x.permute(*permutation)

        return x
