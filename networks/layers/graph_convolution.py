import math

import torch
from torch.nn.parameter import Parameter


class GraphConvolution(torch.nn.Module):
    """
    Implements graph convolutions.
    Adapted from: https://github.com/idiap/potr/blob/main/models/PoseGCN.py
    from the paper 
    "Pose Transformers: Human Motion Prediction with Non-Autoregressive Transformers"
    (2021) by Martínez-González et al.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_nodes: int = 25,
        bias: bool = True
    ) -> None:
        """Constructor.

        The graph convolutions can be defined as \sigma(AxHxW), where A is the 
        adjacency matrix, H is the feature representation from previous layer
        and W is the wegith of the current layer. The dimensions of such martices
        A\in R^{NxN}, H\in R^{NxM} and W\in R^{MxO} where
          - N is the number of nodes
          - M is the number of input features per node
          - O is the number of output features per node

        Args:
          in_features: Number of input features per node.
          out_features: Number of output features per node.
          output_nodes: Number of nodes in the graph.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.n_nodes = n_nodes

        # W\in R^{MxO}
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # A\in R^{NxN}
        self.att = Parameter(torch.FloatTensor(n_nodes, n_nodes))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]

        x_batch = x.view(-1, self.n_nodes, self.in_features)

        # [batch_size, input_dim, output_features]
        # HxW = {NxM}x{MxO} = {NxO}
        support = torch.matmul(x_batch, self.weight)
        # [batch_size, n_nodes, output_features]
        # = {NxN}x{NxO} = {NxO}
        output = torch.matmul(self.att, support)

        if self.bias is not None:
            output = output + self.bias

        # reshape to original shape with
        output = output.view(*x.shape[:-1], self.out_features)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphConvolutionBlock(torch.nn.Module):
    """
    Graph Convolutional Block architecture inspired by: 

    Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
    Transformers (2021) (https://ieeexplore.ieee.org/abstract/document/9607511)
    """

    def __init__(
        self,
        n_nodes: int,
        feature_dim: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.n_nodes = n_nodes

        self.feature_dim = feature_dim

        self.graph_convolution_1 = GraphConvolution(
            n_nodes=self.n_nodes,
            in_features=self.feature_dim,
            out_features=self.feature_dim
        )

        self.batch_norm_1 = torch.nn.BatchNorm1d(
            self.n_nodes * self.feature_dim
        )

        self.graph_convolution_2 = GraphConvolution(
            n_nodes=self.n_nodes,
            in_features=self.feature_dim,
            out_features=self.feature_dim
        )

        self.batch_norm_2 = torch.nn.BatchNorm1d(
            self.n_nodes * self.feature_dim
        )

        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        # input shape: [BATCH, TIMESTEP, JOINT_IDX, FEATURES]

        residual = x

        processed = self.graph_convolution_1(x)

        processed = self.batch_norm_1(
            processed.view(-1, self.n_nodes * self.feature_dim)
        ).view(x.shape)

        processed = self.activation(processed)
        processed = self.dropout(processed)

        processed = self.graph_convolution_2(processed)

        processed = self.batch_norm_2(
            processed.view(-1, self.n_nodes * self.feature_dim)
        ).view(x.shape)

        processed = self.activation(processed)
        processed = self.dropout(processed)

        output = processed + residual

        return output
