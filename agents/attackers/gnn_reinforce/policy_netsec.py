import torch
from typing import Tuple, Dict, List, Optional
from torch_geometric.data import HeteroData

from schemas import NodeGoal
from netsecgame.game_components import GameState, ActionType, IP, Network
from attempt_counts import AttemptCounts
import netaddr

INTENT_IDS = {"discovery": 0, "exploitation": 1, "locate": 2, "exfiltration": 3}


def state_to_pyg(
    state: GameState,
    attempt_counts: Optional[AttemptCounts] = None,
) -> Tuple[HeteroData, Dict[str, dict], Dict[str, list]]:
    """
    Converts the homogenous GameState graph into a PyTorch Geometric HeteroData object.

    Args:
        state: The current NetSecGame observation.
        attempt_counts: Optional per-episode attempt counters. When provided,
            the counters are encoded into host/network/data node features at
            indices documented in
            docs/plans/2026-04-16-per-action-attempt-features-design.md.

    Returns:
        data (HeteroData): The formatted PyG heterogeneous graph.
        object_to_idx (Dict[str, dict]): Maps game objects to their contiguous index
            per node type. Format: {'host': {IP("1.1.1.1"): 0, ...}, ...}
        idx_to_object (Dict[str, list]): Inverse mapping — sorted list of game objects
            per node type. Format: {'host': [IP("1.1.1.1"), ...], ...}
    """
    from node_indices import build_node_indices

    data = HeteroData()
    object_to_idx, idx_to_object, services, datapoints = build_node_indices(state)
    networks = idx_to_object["network"]
    hosts = idx_to_object["host"]
    # services is a list of (Service, host_ip) tuples in index order
    # datapoints is a list of (Data, host_ip) tuples in index order

    def _norm(count: int) -> float:
        return min(float(count), 10.0) / 10.0

    # 1. Map Nodes
    # Networks
    # Precompute known-host counts per network for feat[1] / feat[2].
    all_known_hosts = set(state.known_hosts).union(set(state.controlled_hosts))
    network_host_counts: Dict[Network, int] = {}
    for n in networks:
        try:
            cidr = netaddr.IPNetwork(str(n))
            network_host_counts[n] = sum(1 for h in all_known_hosts if str(h) in cidr)
        except netaddr.AddrFormatError:
            network_host_counts[n] = 0

    if networks:
        net_features = []
        for n in networks:
            feat = torch.zeros(16)
            if attempt_counts is not None:
                feat[0] = _norm(attempt_counts.scan.get(n, 0))
            host_count = network_host_counts[n]
            feat[1] = 1.0 if host_count > 0 else 0.0
            feat[2] = _norm(host_count)
            net_features.append(feat)
        data["network"].x = torch.stack(net_features)
    else:
        data["network"].x = torch.empty((0, 16))

    # Hosts
    host_features = []
    for h in hosts:
        feat = torch.zeros(16)
        if h in state.controlled_hosts:
            feat[0] = 1.0
        try:
            if not netaddr.IPAddress(str(h)).is_private():
                feat[1] = 1.0
        except Exception:
            pass
        if attempt_counts is not None:
            feat[2] = _norm(attempt_counts.findservices.get(h, 0))
            feat[3] = _norm(attempt_counts.finddata.get(h, 0))
            feat[4] = _norm(attempt_counts.exploit.get(h, 0))
        host_features.append(feat)
    data["host"].x = (
        torch.stack(host_features) if host_features else torch.empty((0, 16))
    )

    # Service features
    svc_features = []
    for s, _ in services:
        feat = torch.zeros(16)
        try:
            port = int(s.name.split("/")[0])
            feat[0] = port / 65535.0
        except (ValueError, IndexError):
            pass
        feat[1] = 1.0 if getattr(s, "is_local", False) else 0.0
        svc_features.append(feat)
    data["service"].x = (
        torch.stack(svc_features) if svc_features else torch.empty((0, 16))
    )

    # Data (unique data objects; first host wins if duplicate)
    # Precompute: which data items are already exfiltrated (on a controlled public host)
    controlled_public_hosts = set(
        h for h in state.controlled_hosts if not netaddr.IPAddress(str(h)).is_private()
    )
    exfiltrated_data = set()
    for h in controlled_public_hosts:
        for d in state.known_data.get(h, []):
            exfiltrated_data.add(d)

    data_features = []
    for d, host_ip in datapoints:
        feat = torch.zeros(16)
        feat[0] = 1.0 if d in exfiltrated_data else 0.0
        try:
            feat[1] = 0.0 if netaddr.IPAddress(str(host_ip)).is_private() else 1.0
        except Exception:
            pass
        if attempt_counts is not None:
            feat[2] = _norm(attempt_counts.exfil.get(d, 0))
        data_features.append(feat)
    data["data"].x = (
        torch.stack(data_features) if data_features else torch.empty((0, 16))
    )

    # 2. Map Edges
    # Host -> Network
    host_net_edges = []
    for host_idx, host in enumerate(hosts):
        for net_idx, net in enumerate(networks):
            try:
                if str(host) in netaddr.IPNetwork(str(net)):
                    host_net_edges.append([host_idx, net_idx])
            except netaddr.AddrFormatError:
                pass
    data["host", "in", "network"].edge_index = (
        torch.tensor(host_net_edges, dtype=torch.long).t()
        if host_net_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    data["network", "contains", "host"].edge_index = (
        torch.tensor([[dst, src] for src, dst in host_net_edges], dtype=torch.long).t()
        if host_net_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    # Host -> Service (+ reverse for message flow back to host)
    host_svc_edges = []
    for svc_idx, (s, host_ip) in enumerate(services):
        if host_ip in object_to_idx["host"]:
            host_svc_edges.append([object_to_idx["host"][host_ip], svc_idx])
    data["host", "runs", "service"].edge_index = (
        torch.tensor(host_svc_edges, dtype=torch.long).t()
        if host_svc_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    data["service", "rev_runs", "host"].edge_index = (
        torch.tensor([[dst, src] for src, dst in host_svc_edges], dtype=torch.long).t()
        if host_svc_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    # Host -> Data (+ reverse for message flow back to host)
    host_data_edges = []
    for data_idx, (d, host_ip) in enumerate(datapoints):
        if host_ip in object_to_idx["host"]:
            host_data_edges.append([object_to_idx["host"][host_ip], data_idx])
    data["host", "stores", "data"].edge_index = (
        torch.tensor(host_data_edges, dtype=torch.long).t()
        if host_data_edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    data["data", "rev_stores", "host"].edge_index = (
        torch.tensor([[dst, src] for src, dst in host_data_edges], dtype=torch.long).t()
        if host_data_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    return data, object_to_idx, idx_to_object


# Map ActionType Enums to a continuous embedded space
ACTION_TYPE_IDS = {
    ActionType.ScanNetwork: 0,
    ActionType.FindServices: 1,
    ActionType.FindData: 2,
    ActionType.ExploitService: 3,
    ActionType.ExfiltrateData: 4,
}

# Sorted list of ActionType values (index matches ACTION_TYPE_IDS integer)
ACTION_TYPE_LIST: List[ActionType] = sorted(
    ACTION_TYPE_IDS.keys(), key=lambda at: ACTION_TYPE_IDS[at]
)

# Primary target parameter key for each action type (step 3)
ACTION_TARGET_PARAM: Dict[ActionType, str] = {
    ActionType.ScanNetwork: "target_network",
    ActionType.FindServices: "target_host",
    ActionType.ExploitService: "target_host",
    ActionType.FindData: "target_host",
    ActionType.ExfiltrateData: "data",
}

# PyG node type that holds the primary target for each action type (step 3)
ACTION_TARGET_NODE_TYPE: Dict[ActionType, str] = {
    ActionType.ScanNetwork: "network",
    ActionType.FindServices: "host",
    ActionType.ExploitService: "host",
    ActionType.FindData: "host",
    ActionType.ExfiltrateData: "data",
}

# Secondary parameter for actions that require a fourth decision (step 4)
ACTION_SECONDARY_PARAM: Dict[ActionType, str] = {
    ActionType.ExploitService: "target_service",
    ActionType.ExfiltrateData: "target_host",
}

ACTION_SECONDARY_NODE_TYPE: Dict[ActionType, str] = {
    ActionType.ExploitService: "service",
    ActionType.ExfiltrateData: "host",
}
