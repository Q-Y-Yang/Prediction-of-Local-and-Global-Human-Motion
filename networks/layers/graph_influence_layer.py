import torch
import torch.nn as nn


def k_hop_neighbourhood_matrix(adjacency_matrix, k_hops=2):
    id_matrix = torch.eye(
        adjacency_matrix.size(0),
        device=adjacency_matrix.device, dtype=adjacency_matrix.dtype
    )

    one_hop_matrix = adjacency_matrix + id_matrix
    k_hop_matrix = torch.linalg.matrix_power(one_hop_matrix, k_hops)
    k_hop_matrix = torch.clamp(k_hop_matrix, 0, 1)

    return k_hop_matrix


def find_max_k_hops(adjacency_matrix):
    i = 0
    old_extended_matrix = torch.zeros_like(adjacency_matrix)
    new_extended_matrix = adjacency_matrix.clone()
    while not torch.equal(old_extended_matrix, new_extended_matrix):
        i += 1
        old_extended_matrix = new_extended_matrix.clone()
        new_extended_matrix = k_hop_neighbourhood_matrix(
            adjacency_matrix, k_hops=i
        )
    return i - 1


def compute_distance_matrix(adjacency_matrix, k_hops=None):
    # Initialize the distance matrix as zeros
    distance_matrix = torch.zeros_like(adjacency_matrix, dtype=torch.float)

    if k_hops is None:
        k_hops = find_max_k_hops(adjacency_matrix) + 1

    if k_hops < 0:
        raise ValueError(f"k_hops needs to be >= 0, but is {k_hops}")

    for steps in range(1, k_hops + 1):
        k_hop_matrix = k_hop_neighbourhood_matrix(adjacency_matrix, steps)
        distance_matrix += k_hop_matrix

    # Adjust the distance matrix to set the correct distances
    max_distance = torch.max(distance_matrix)
    distance_matrix = max_distance - distance_matrix
    distance_matrix[distance_matrix >= max_distance] = float('inf')

    return distance_matrix


def compute_discount_matrix(distance_matrix, discount_factor=0.7):
    with torch.no_grad():
        discount_matrix = torch.pow(discount_factor, distance_matrix)
    return discount_matrix


class GraphInfluenceLayer(nn.Module):
    '''
    Computes the Hardamad product (element-wise multiplication) 
    of a trainable weight matrix and a neighborhood matrix.
    The result is then multiplied with the input data.
    e.g. (weights * neighbourhood) @ input
    When k_hops is not None, then only the k-neighbourhood is used.
    When discount_factor is not None, then each step is discounted by the factor,
    so the effect is weaker the further the neighbour is away.

    The trainable weight matrix has the potential to measure the interactive influence
    between graph nodes.

    Models the interactive influence between graph nodes
    by computing the Hadamard product (element-wise multiplication) of a trainable 
    weight matrix and a neighborhood matrix, and then multiplying the result
    with the input data. 
    This layer uses the adjacency matrix of the graph to influence the transformation
    of node features based on their connectivity, optionally considering only 
    a k-hop neighborhood and applying a discount factor to model
    the diminishing influence of distant nodes.
    '''

    def __init__(self, k_hops, discount_factor, adjacency_matrix, feature_dim=3,
                 bias: bool = True):

        super(GraphInfluenceLayer, self).__init__()

        self.k_hops = k_hops
        self.discount_factor = discount_factor

        # initialize/adjust the neighbourhood matrix based on k_hops and discount_factor
        neighbourhood_matrix = self.make_neighbourhood_matrix(adjacency_matrix)
        self.register_buffer('neighbourhood_matrix', neighbourhood_matrix)

        # Initialize weight and bias parameters
        # Initialize weights based on the neighbourhood matrix
        self.weight = nn.Parameter(
            torch.Tensor(self.neighbourhood_matrix.size())
        )

        if bias:
            # self.bias = nn.Parameter(torch.Tensor(self.neighbourhood_matrix.size(0)))
            self.bias = nn.Parameter(torch.Tensor(
                neighbourhood_matrix.size(0), feature_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()  # Reset parameters using suitable initialization

    def make_neighbourhood_matrix(self, adjacency_matrix):
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float)
        # Apply k-hops and/or discount factor adjustments if necessary
        if self.k_hops is not None:
            neighbourhood_matrix = k_hop_neighbourhood_matrix(
                adjacency_matrix, self.k_hops
            )
        if self.discount_factor is not None:
            distance_matrix = compute_distance_matrix(
                adjacency_matrix, k_hops=self.k_hops
            )
            neighbourhood_matrix = compute_discount_matrix(
                distance_matrix, self.discount_factor
            )

        if not isinstance(neighbourhood_matrix, torch.Tensor):
            neighbourhood_matrix = torch.tensor(
                neighbourhood_matrix, dtype=torch.float)

        else:
            neighbourhood_matrix = (neighbourhood_matrix
                                    .clone()
                                    .detach()
                                    .requires_grad_(True))

        return neighbourhood_matrix

    def forward(self, x):

        # Validate input dimensions and types
        if not isinstance(self.neighbourhood_matrix, torch.Tensor):
            raise TypeError("Neighbourhood matrix must be a PyTorch tensor")

        # Use the cached neighbourhood matrix
        neighbourhood_matrix = self.neighbourhood_matrix

        # Compute the Hadamard product of the weight matrix and the neighbourhood matrix
        weighted_neighbourhood = self.weight * neighbourhood_matrix

        # Perform the matrix multiplication with the input data 'x'
        output = torch.matmul(weighted_neighbourhood, x)

        # Add bias if it exists
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output)

        return output

    def reset_parameters(self):
        # Initialize the parameters, for example, using Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
