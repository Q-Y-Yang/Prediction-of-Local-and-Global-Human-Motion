import torch

from networks.layers import (
    JointSeparateLinear,
    SpatialAttention,
    GraphConvolutionBlock
)


class PoseEncoder(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_joints: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.n_joints = n_joints

        self.missing_joint_token = torch.nn.Parameter(
            torch.FloatTensor(torch.randn(self.in_features))
        )

        self.projection_linear = JointSeparateLinear(
            n_joints=self.n_joints,
            in_features=self.in_features,
            out_features=self.out_features
        )

        self.spatial_attention = SpatialAttention(
            n_joints=self.n_joints,
            joint_embedding_dim=self.out_features,
            dropout=dropout
        )

    def replace_missing_values(self, x: torch.Tensor):
        mask = torch.all(torch.isnan(x), dim=-1, keepdim=True)
        mask = mask.expand_as(x)
        x = torch.where(mask, self.missing_joint_token, x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]

        processed = self.replace_missing_values(x)

        # [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        # -> [BATCH, TIMESTEP, JOINT_IDX, EMBEDDING_DIM]
        processed = self.projection_linear(processed)
        output = self.spatial_attention(processed)

        return output


class PoseEncoderGCN(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_joints: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.n_joints = n_joints

        self.embedding_linear = JointSeparateLinear(
            n_joints=self.n_joints,
            in_features=self.in_features,
            out_features=self.out_features
        )

        self.missing_joint_token = torch.nn.Parameter(
            torch.FloatTensor(torch.randn(self.in_features))
        )

        self.graph_convolution_block = GraphConvolutionBlock(
            n_nodes=self.n_joints,
            feature_dim=self.out_features,
            dropout=dropout
        )

        self.dropout = torch.nn.Dropout(dropout)

    def replace_missing_values(self, x: torch.Tensor):
        mask = torch.all(torch.isnan(x), dim=-1, keepdim=True)
        mask = mask.expand_as(x)
        x = torch.where(mask, self.missing_joint_token, x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]

        processed = self.replace_missing_values(x)
        # print(x)
        # [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        # -> [BATCH, TIMESTEP, JOINT_IDX, EMBEDDING_DIM]
        processed = self.embedding_linear(processed)
        output = self.graph_convolution_block(processed)

        return output


class PoseDecoder(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_joints: int
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.n_joints = n_joints

        self.projection_linear = JointSeparateLinear(
            n_joints=self.n_joints,
            in_features=self.in_features,
            out_features=self.out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, EMBEDDING_DIM]

        # [BATCH, TIMESTEP, JOINT_IDX, EMBEDDING_DIM]
        # -> [BATCH, TIMESTEP, JOINT_IDX, FEATURES]
        output = self.projection_linear(x)

        return output
