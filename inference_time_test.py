import torch

from networks import (BaselineTransformer, GCNTemporalTransformer,
                      SpatioTemporalTransformer)
from networks.layers import GraphConvolutionBlock, SpatialAttention

DEVICE = "cuda"


def measure_runtime(module, sample_input: dict[str, torch.Tensor], num_runs=100):
    """
    Measures the average runtime of a PyTorch module across multiple runs.

    Arguments:
        module -- The PyTorch module to evaluate.
        sample_input -- A sample input tensor for the module.

    Keyword Arguments:
        num_runs -- Number iterations to average. (default: {1000})

    Returns:
        Average runtime in milliseconds.
    """
    module = module.to(DEVICE)
    module.eval()
    sample_input = {key: t.to(DEVICE) for key, t in sample_input.items()}

    # cuda time events to measure time between start and end
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # cold pre-runs
        for _ in range(10):
            _ = module(**sample_input)

        # measured runs
        start_event.record()
        for _ in range(num_runs):
            _ = module(**sample_input)
        end_event.record()

    # needed to wait for asynchronous PyTorch operations
    torch.cuda.synchronize()
    # time in ms
    elapsed_time = start_event.elapsed_time(end_event) / num_runs

    return elapsed_time


if __name__ == "__main__":
    sequence_length = 30
    n_joints = 25
    joint_feature_dim = 3

    history = torch.randn(1, sequence_length, n_joints, joint_feature_dim)
    noisy = torch.randn(1, sequence_length, n_joints, joint_feature_dim)
    sample_input = {
        "history": history,
        "noisy": noisy
    }

    print("Measuring Spatio-Temporal Transformer")
    model = SpatioTemporalTransformer(
        n_joints=n_joints,
        joint_feature_dim=joint_feature_dim,
        joint_embedding_dim=64,
        max_length=sequence_length,
        n_attention_blocks=3
    )
    st_ms = measure_runtime(model, sample_input)
    print(f"Spatio-Temporal {st_ms:.3f}ms")

    print("Measuring GCN-Temporal Transformer")
    model = GCNTemporalTransformer(
        n_joints=n_joints,
        joint_feature_dim=joint_feature_dim,
        joint_embedding_dim=64,
        max_length=sequence_length,
        n_attention_blocks=3
    )
    gcn_ms = measure_runtime(model, sample_input)
    print(f"GCN-Temporal {gcn_ms:.3f}ms")

    print("Measuring Baseline Transformer")
    model = BaselineTransformer(
        n_joints=n_joints,
        joint_feature_dim=3,
        embedding_dim=128,
        n_transformer_blocks=6,
        max_length=sequence_length,
    )
    base_ms = measure_runtime(model, sample_input)
    print(f"Base {base_ms:.3f}ms")

    sample_input = {
        "input_sequence": torch.randn(1, 25, 25, 64),
    }

    print("Measuring Spatial Attention")
    spatial_attention = SpatialAttention(n_joints=25, joint_embedding_dim=64)
    print("SpatialAttention: ", measure_runtime(
        spatial_attention, sample_input, num_runs=1000))

    sample_input = {
        "x": torch.randn(1, 25, 25, 64),
    }

    print("Measuring GCN-Network")
    gcn_block = GraphConvolutionBlock(n_nodes=25, feature_dim=64)
    print("GCN: ", measure_runtime(gcn_block, sample_input, num_runs=1000))
