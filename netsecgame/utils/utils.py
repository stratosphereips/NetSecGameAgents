from __future__ import annotations

import hashlib
import json
import os

from ..game_components import (
    Action,
    ActionType,
    Data,
    GameState,
    IP,
    Network,
    Observation,
    Service,
)


def get_file_hash(filepath: str, hash_func: str = "sha256", chunk_size: int = 4096) -> str:
    hash_algorithm = hashlib.new(hash_func)
    with open(filepath, "rb") as file_handle:
        chunk = file_handle.read(chunk_size)
        while chunk:
            hash_algorithm.update(chunk)
            chunk = file_handle.read(chunk_size)
    return hash_algorithm.hexdigest()


def state_as_ordered_string(state: GameState) -> str:
    ret = ""
    ret += f"nets:[{','.join([str(x) for x in sorted(state.known_networks)])}],"
    ret += f"hosts:[{','.join([str(x) for x in sorted(state.known_hosts)])}],"
    ret += f"controlled:[{','.join([str(x) for x in sorted(state.controlled_hosts)])}],"
    ret += "services:{"
    for host in sorted(state.known_services.keys()):
        ret += f"{host}:[{','.join([str(x) for x in sorted(state.known_services[host])])}]"
    ret += "},data:{"
    for host in sorted(state.known_data.keys()):
        ret += f"{host}:[{','.join([str(x) for x in sorted(state.known_data[host])])}]"
    ret += "}, blocks:{"
    for host in sorted(state.known_blocks.keys()):
        ret += f"{host}:[{','.join([str(x) for x in sorted(state.known_blocks[host])])}]"
    ret += "}"
    return ret


def observation_as_dict(observation: Observation) -> dict:
    return {
        "state": observation.state.as_dict,
        "reward": observation.reward,
        "end": observation.end,
        "info": dict(observation.info),
    }


def observation_to_str(observation: Observation) -> str:
    return json.dumps(observation_as_dict(observation))


def observation_from_dict(data: dict) -> Observation:
    return Observation(
        state=GameState.from_dict(data.get("state", {})),
        reward=float(data.get("reward", 0.0)),
        end=bool(data.get("end", False)),
        info=data.get("info", {}),
    )


def observation_from_str(json_str: str) -> Observation:
    return observation_from_dict(json.loads(json_str))


def store_trajectories_to_jsonl(trajectories: list, directory: str, filename: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, f"{filename.rstrip('jsonl')}.jsonl")
    try:
        import jsonlines
    except ImportError:
        with open(filepath, "a", encoding="utf-8") as writer:
            writer.write(json.dumps(trajectories))
            writer.write("\n")
        return

    with jsonlines.open(filepath, "a") as writer:
        writer.write(trajectories)


def read_trajectories_from_jsonl(filepath: str) -> list:
    raise NotImplementedError("This function is not yet implemented.")


def generate_valid_actions(state: GameState, include_blocks: bool = False) -> list:
    valid_actions = set()

    def is_fw_blocked(current_state: GameState, src_ip, dst_ip) -> bool:
        blocked = False
        try:
            blocked = dst_ip in current_state.known_blocks[src_ip]
        except KeyError:
            pass
        return blocked

    for source_host in state.controlled_hosts:
        for network in state.known_networks:
            valid_actions.add(
                Action(
                    ActionType.ScanNetwork,
                    parameters={"target_network": network, "source_host": source_host},
                )
            )

        for blocked_host in state.known_hosts:
            if not is_fw_blocked(state, source_host, blocked_host):
                valid_actions.add(
                    Action(
                        ActionType.FindServices,
                        parameters={"target_host": blocked_host, "source_host": source_host},
                    )
                )

        for blocked_host, service_list in state.known_services.items():
            if not is_fw_blocked(state, source_host, blocked_host):
                for service in service_list:
                    valid_actions.add(
                        Action(
                            ActionType.ExploitService,
                            parameters={
                                "target_host": blocked_host,
                                "target_service": service,
                                "source_host": source_host,
                            },
                        )
                    )

        for blocked_host in state.controlled_hosts:
            if not is_fw_blocked(state, source_host, blocked_host):
                valid_actions.add(
                    Action(
                        ActionType.FindData,
                        parameters={"target_host": blocked_host, "source_host": blocked_host},
                    )
                )

        for data_source_host, data_list in state.known_data.items():
            for data in data_list:
                for target_host in state.controlled_hosts:
                    if target_host != data_source_host and not is_fw_blocked(state, data_source_host, target_host):
                        valid_actions.add(
                            Action(
                                ActionType.ExfiltrateData,
                                parameters={
                                    "target_host": target_host,
                                    "source_host": data_source_host,
                                    "data": data,
                                },
                            )
                        )

        if include_blocks:
            for block_source_host in state.controlled_hosts:
                for target_host in state.controlled_hosts:
                    if not is_fw_blocked(state, block_source_host, target_host):
                        for blocked_host in state.known_hosts:
                            valid_actions.add(
                                Action(
                                    ActionType.BlockIP,
                                    {
                                        "target_host": target_host,
                                        "source_host": block_source_host,
                                        "blocked_host": blocked_host,
                                    },
                                )
                            )

    return list(valid_actions)
