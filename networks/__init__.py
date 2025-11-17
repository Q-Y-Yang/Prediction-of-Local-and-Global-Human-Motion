from enum import Enum

import torch

from networks.baseline_transformer import (BaselineTransformer,
                                           GraphInfluenceTransformer)
from networks.gcn_temporal import GCNTemporalTransformer
from networks.spatio_temporal import SpatioTemporalTransformer


class NetworkType(Enum):
    BASELINE_TRANSFORMER = "baseline-transformer"
    GRAPH_INFLUENCE_TRANSFORMER = "graph-influence-transformer"
    SPATIO_TEMPORAL = "spatio-temporal"
    GCN_TEMPORAL = "gcn-temporal"


def load_model_snapshot(
    network_type: NetworkType,
    filepath: str
) -> torch.nn.Module | None:
    match network_type:
        case NetworkType.BASELINE_TRANSFORMER:
            return BaselineTransformer.load_model(filepath)
        case NetworkType.GRAPH_INFLUENCE_TRANSFORMER:
            return GraphInfluenceTransformer.load_model(filepath)
        case NetworkType.SPATIO_TEMPORAL:
            return SpatioTemporalTransformer.load_model(filepath)
        case NetworkType.GCN_TEMPORAL:
            return GCNTemporalTransformer.load_model(filepath)
        case _:
            return None


__all__ = [
    "BaselineTransformer",
    "GraphInfluenceTransformer",
    "SpatioTemporalTransformer",
    "GCNTemporalTransformer"
]
