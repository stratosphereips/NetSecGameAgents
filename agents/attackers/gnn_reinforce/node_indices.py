"""Shared deterministic node-index construction.

Single source of truth for the sort order and keying convention used by
state_to_pyg (to lay out HeteroData tensors) and ScriptedPlanner (to emit
target_node_key strings that ground back to the same game objects).
"""
from typing import Any, Dict, List, Tuple
from netsecgame.game_components import GameState


def build_node_indices(
    state: GameState,
) -> Tuple[
    Dict[str, Dict[Any, int]],  # object_to_idx (game object → contiguous int index)
    Dict[str, list],            # idx_to_object (list of game objects per type)
    List[Tuple],                # services: [(Service, host_ip), ...] in index order
    List[Tuple],                # datapoints: [(Data, host_ip), ...] in index order
]:
    """Construct deterministic (object_to_idx, idx_to_object, services, datapoints) mappings.

    Node types: 'network', 'host', 'service', 'data'.

    Service keys in object_to_idx are Service objects (deduplicated — first host
    in sorted order wins when the same Service appears on multiple hosts).
    idx_to_object['service'] is a list of Service objects in the same order.

    The third return value, services, is a list of (Service, host_ip) tuples in
    index order, needed by state_to_pyg to construct host→service edges.

    Data is deduplicated by Data identity; first-seen host order wins.
    The fourth return value, datapoints, is a list of (Data, host_ip) tuples in
    index order, needed by state_to_pyg to construct host→data edges and features.
    """
    object_to_idx: Dict[str, dict] = {
        'network': {}, 'host': {}, 'service': {}, 'data': {},
    }

    # Networks
    networks = sorted(list(state.known_networks), key=lambda x: str(x))
    for idx, net in enumerate(networks):
        object_to_idx['network'][net] = idx

    # Hosts (union of known + controlled)
    hosts = sorted(
        list(set(state.known_hosts).union(set(state.controlled_hosts))),
        key=lambda x: str(x),
    )
    for idx, host in enumerate(hosts):
        object_to_idx['host'][host] = idx

    # Services — deduplicated by Service object; first host (sorted) wins
    services: List[Tuple] = []  # list of (Service, host_ip); position = node index
    for host_ip in sorted(state.known_services.keys(), key=lambda x: str(x)):
        for s in sorted(list(state.known_services[host_ip]), key=lambda x: str(x)):
            if s not in object_to_idx['service']:
                object_to_idx['service'][s] = len(services)
                services.append((s, host_ip))

    # Data — deduplicated by Data identity; first host (sorted) wins
    datapoints: List[Tuple] = []  # list of (Data, host_ip); position = node index
    for host_ip in sorted(state.known_data.keys(), key=lambda x: str(x)):
        for d in sorted(list(state.known_data[host_ip]), key=lambda x: str(x)):
            if d not in object_to_idx['data']:
                object_to_idx['data'][d] = len(datapoints)
                datapoints.append((d, host_ip))

    idx_to_object = {
        'network': networks,
        'host': hosts,
        'service': [s for s, _ in services],
        'data': [d for d, _ in datapoints],
    }

    return object_to_idx, idx_to_object, services, datapoints
