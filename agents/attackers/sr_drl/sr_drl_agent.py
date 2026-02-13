from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from os import makedirs, path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency error path
    raise ImportError(
        "PyTorch is required for the SR-DRL agent. "
        "Install extras with:\n\n"
        "  pip install -e .[sr_drl]\n"
    ) from exc

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None  # type: ignore

from netsecgame import (
    Action,
    ActionType,
    AgentRole,
    BaseAgent,
    GameState,
    Observation,
    generate_valid_actions,
)
from netsecgame.game_components import AgentStatus, IP, Network, Service, Data

from NetSecGameAgents.agents.attackers.sr_drl.net import GraphPolicyNet
from NetSecGameAgents.agents.attackers.sr_drl.rl import a2c


NODE_TYPE_NETWORK = 0
NODE_TYPE_KNOWN_HOST = 1
NODE_TYPE_CONTROLLED_HOST = 2
NODE_TYPE_SERVICE = 3
NODE_TYPE_DATA = 4
NUM_NODE_TYPES = 5


@dataclass
class SRDRLConfig:
    gamma: float = 0.99
    alpha_v: float = 0.1
    alpha_h: float = 0.1
    q_range: Tuple[float, float] | None = None
    lr: float = 3e-3
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 3.0
    mp_iterations: int = 5
    seed: int = 42


def build_graph_from_state(state: GameState) -> Tuple[np.ndarray, np.ndarray, List[object]]:
    """
    Convert GameState into a graph representation similar to graph_agent_utils.state_as_graph,
    but also return the list of original entities per node index.

    Returns:
        node_features: (N, NUM_NODE_TYPES) array
        edge_index: (2, E) array of integer indices
        entities: list of original objects (Network, IP, Service, Data)
    """
    node_features = []
    entities: List[object] = []
    edge_list: List[Tuple[int, int]] = []

    # Networks
    for net in state.known_networks:
        idx = len(entities)
        entities.append(net)
        feat = np.zeros(NUM_NODE_TYPES, dtype=np.float32)
        feat[NODE_TYPE_NETWORK] = 1.0
        node_features.append(feat)

    # Hosts
    for host in state.known_hosts:
        idx = len(entities)
        entities.append(host)
        feat = np.zeros(NUM_NODE_TYPES, dtype=np.float32)
        if host in state.controlled_hosts:
            feat[NODE_TYPE_CONTROLLED_HOST] = 1.0
        else:
            feat[NODE_TYPE_KNOWN_HOST] = 1.0
        node_features.append(feat)

    # Services
    for host, services in state.known_services.items():
        for svc in services:
            idx = len(entities)
            entities.append(svc)
            feat = np.zeros(NUM_NODE_TYPES, dtype=np.float32)
            feat[NODE_TYPE_SERVICE] = 1.0
            node_features.append(feat)

    # Data
    for host, data_list in state.known_data.items():
        for d in data_list:
            idx = len(entities)
            entities.append(d)
            feat = np.zeros(NUM_NODE_TYPES, dtype=np.float32)
            feat[NODE_TYPE_DATA] = 1.0
            node_features.append(feat)

    # Build mapping from entity -> node index for edges
    entity_to_idx: Dict[object, int] = {ent: i for i, ent in enumerate(entities)}

    # Edges: network <-> host
    for host in state.known_hosts:
        host_idx = entity_to_idx[host]
        for net in state.known_networks:
            try:
                if IP(str(host)) in Network(str(net.ip), net.mask):
                    net_idx = entity_to_idx[net]
                    edge_list.append((net_idx, host_idx))
                    edge_list.append((host_idx, net_idx))
            except Exception:
                # Best-effort; if IP/network conversion fails, skip
                continue

    # Edges: host <-> service
    for host, services in state.known_services.items():
        host_idx = entity_to_idx.get(host)
        if host_idx is None:
            continue
        for svc in services:
            svc_idx = entity_to_idx.get(svc)
            if svc_idx is None:
                continue
            edge_list.append((host_idx, svc_idx))
            edge_list.append((svc_idx, host_idx))

    # Edges: host <-> data
    for host, data_list in state.known_data.items():
        host_idx = entity_to_idx.get(host)
        if host_idx is None:
            continue
        for d in data_list:
            d_idx = entity_to_idx.get(d)
            if d_idx is None:
                continue
            edge_list.append((host_idx, d_idx))
            edge_list.append((d_idx, host_idx))

    node_features_arr = np.asarray(node_features, dtype=np.float32)
    if edge_list:
        edge_index_arr = np.asarray(edge_list, dtype=np.int64).T
    else:
        edge_index_arr = np.zeros((2, 0), dtype=np.int64)

    return node_features_arr, edge_index_arr, entities


def build_action_mapping(
    state: GameState,
    entities: List[object],
    logger: logging.Logger | None = None,
) -> Tuple[List[int], Dict[int, Action]]:
    """
    Build mapping from graph node indices to concrete Actions.

    Strategy:
        - use generate_valid_actions(state) to get valid Actions
        - group actions by an "anchor" entity (Network/IP/Service/Data)
        - for each anchor node, select a representative action according
          to a fixed priority ordering so that the policy chooses nodes,
          and the agent derives concrete actions from them.
    """
    valid_actions = list(generate_valid_actions(state))

    node_actions: Dict[int, List[Action]] = {}
    entity_to_idx: Dict[object, int] = {ent: i for i, ent in enumerate(entities)}

    for action in valid_actions:
        anchor: object | None = None
        if action.type == ActionType.ScanNetwork:
            anchor = action.parameters.get("target_network")
        elif action.type == ActionType.FindServices:
            anchor = action.parameters.get("target_host")
        elif action.type == ActionType.FindData:
            anchor = action.parameters.get("target_host")
        elif action.type == ActionType.ExploitService:
            anchor = action.parameters.get("target_service") or action.parameters.get(
                "target_host"
            )
        elif action.type == ActionType.ExfiltrateData:
            anchor = action.parameters.get("data") or action.parameters.get(
                "target_host"
            )

        if anchor is None:
            continue
        idx = entity_to_idx.get(anchor)
        if idx is None:
            continue
        node_actions.setdefault(idx, []).append(action)

    actionable_nodes = sorted(node_actions.keys())
    node_to_action: Dict[int, Action] = {}

    # Fixed priority over action types for a given node
    type_priority = {
        ActionType.ExfiltrateData: 0,
        ActionType.ExploitService: 1,
        ActionType.FindData: 2,
        ActionType.FindServices: 3,
        ActionType.ScanNetwork: 4,
    }

    for node_idx in actionable_nodes:
        actions = node_actions[node_idx]
        actions.sort(key=lambda a: (type_priority.get(a.type, 99), str(a)))
        node_to_action[node_idx] = actions[0]

    if logger:
        logger.debug(
            "SR-DRL: %d actionable nodes, %d valid actions",
            len(actionable_nodes),
            len(valid_actions),
        )

    return actionable_nodes, node_to_action


class SRDRLAttackerAgent(BaseAgent):
    """
    Graph-based attacker agent for NetSecGame using a SR-DRL-inspired
    message-passing policy with an A2C loss.
    """

    def __init__(
        self,
        host: str,
        port: int,
        role: AgentRole = AgentRole.Attacker,
        config: SRDRLConfig | None = None,
        logdir: str | None = None,
        capture_trajectories: bool = False,
    ) -> None:
        super().__init__(host, port, role)
        self.config = config or SRDRLConfig()
        self._registered = False
        self.capture_trajectories = capture_trajectories
        self.trajectories: List[str] = []

        self._seed_everything(self.config.seed)

        self._logger.setLevel(logging.INFO)
        self.logdir = logdir or path.join(
            path.dirname(path.abspath(__file__)), "logs"
        )
        if not path.exists(self.logdir):
            makedirs(self.logdir, exist_ok=True)

        # Node features are simple one-hot type indicators
        self.net = GraphPolicyNet(
            num_node_features=NUM_NODE_TYPES,
            mp_iterations=self.config.mp_iterations,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
        )

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Override to optionally capture trajectories on reset
    def request_game_reset(
        self,
        request_trajectory: bool = False,
        randomize_topology: bool = True,
        randomize_topology_seed: int | None = None,
    ) -> Observation | None:
        """
        Wrapper around BaseAgent.request_game_reset that, when
        capture_trajectories is enabled, requests the last episode
        trajectory and stores it in self.trajectories.
        """
        effective_req_traj = request_trajectory or self.capture_trajectories
        obs = super().request_game_reset(
            request_trajectory=effective_req_traj,
            randomize_topology=randomize_topology,
            randomize_topology_seed=randomize_topology_seed,
        )
        if (
            self.capture_trajectories
            and obs is not None
            and getattr(obs, "info", None)
            and "last_trajectory" in obs.info
        ):
            try:
                self.trajectories.append(
                    json.dumps(obs.info["last_trajectory"]) + "\n"
                )
            except TypeError:
                # If the trajectory is not JSON-serializable for some reason,
                # skip it rather than failing the whole run.
                self.logger.warning(
                    "SR-DRL: failed to serialize last_trajectory for storage."
                )
        return obs

    def recompute_reward(self, observation: Observation) -> Observation:
        """
        Reward shaping similar to Q-learning / conceptual agent.
        """
        state = observation.state
        end = observation.end
        info = observation.info
        reward = observation.reward

        if info and info.get("end_reason") == AgentStatus.Fail:
            reward = -1000
        elif info and info.get("end_reason") == AgentStatus.Success:
            reward = 1000
        elif info and info.get("end_reason") == AgentStatus.TimeoutReached:
            reward = -100
        else:
            reward = -1

        return Observation(state, reward, end, info)

    def encode_state(
        self, state: GameState
    ) -> Tuple[torch.Tensor, torch.Tensor, List[object]]:
        node_feats_np, edge_index_np, entities = build_graph_from_state(state)
        node_feats = torch.from_numpy(node_feats_np)
        edge_index = torch.from_numpy(edge_index_np)
        return node_feats, edge_index, entities

    def select_action(
        self,
        observation: Observation,
    ) -> Tuple[Action | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Given an Observation, select an Action using the graph policy.

        Returns:
            action (or None if no valid actions),
            value (scalar),
            pi (probability of selected action),
            log_pi (for diagnostics)
        """
        state = observation.state
        node_feats, edge_index, entities = self.encode_state(state)
        actionable_nodes, node_to_action = build_action_mapping(
            state, entities, self.logger
        )

        if not actionable_nodes:
            self.logger.info("SR-DRL: no actionable nodes, skipping step.")
            return None, 0.0, 1.0, 0.0

        sel_idx_tensor, value_tensor, pi_tensor = self.net(
            node_feats, edge_index, actionable_nodes
        )
        sel_idx = int(sel_idx_tensor[0].item())
        action = node_to_action.get(sel_idx)
        if action is None:
            self.logger.warning(
                "SR-DRL: selected node %d has no mapped action, skipping.", sel_idx
            )
            return None, None, None

        self.logger.info(
            "SR-DRL: selected node %d -> action %s", sel_idx, action
        )
        return action, value_tensor, pi_tensor

    def play_episode(self, training: bool = True) -> Tuple[float, int, AgentStatus | None]:
        """
        Play a single episode using SR-DRL policy. If training=True,
        update the network online with A2C after each step.

        Returns:
            total_return, num_steps
        """
        # First episode: register; subsequent ones: request game reset.
        if not self._registered:
            observation = self.register()
            self._registered = True
        else:
            observation = self.request_game_reset()

        if observation is None:
            self.logger.error(
                "SR-DRL: failed to obtain initial observation (register/reset)."
            )
            return 0.0, 0, None

        observation = self.recompute_reward(observation)

        total_return = 0.0
        num_steps = 0
        done = observation.end

        last_value = 0.0

        while not done:
            num_steps += 1

            action, value_tensor, pi_tensor = self.select_action(observation)
            if action is None:
                # No valid action; we request game reset to avoid deadlock.
                self.logger.warning(
                    "SR-DRL: no action selected, requesting game reset."
                )
                observation = self.request_game_reset()
                observation = self.recompute_reward(observation)
                done = observation.end
                continue

            next_obs = self.make_step(action)
            next_obs = self.recompute_reward(next_obs)

            reward = float(next_obs.reward)
            total_return += reward
            done = next_obs.end

            # Bootstrap value for next state
            next_value_tensor = torch.zeros_like(value_tensor)
            if not done:
                node_feats, edge_index, entities = self.encode_state(next_obs.state)
                actionable_nodes, _ = build_action_mapping(
                    next_obs.state, entities, self.logger
                )
                if actionable_nodes:
                    _, v_next, _ = self.net(
                        node_feats, edge_index, actionable_nodes, only_v=True
                    )
                    next_value_tensor = v_next

            if training:
                # Scalars for logging
                value = float(value_tensor[0, 0].item())
                next_value = float(next_value_tensor[0, 0].item())

                # Tensors wired to the network graph for loss
                r_tensor = torch.tensor(
                    [reward],
                    dtype=torch.float32,
                    device=value_tensor.device,
                )
                v_tensor = value_tensor.view(-1)
                v_next_tensor = next_value_tensor.view(-1)
                pi_tensor_flat = pi_tensor.view(-1)

                loss, loss_pi, loss_v, loss_h, entropy = a2c(
                    r_tensor,
                    v_tensor,
                    v_next_tensor,
                    pi_tensor_flat,
                    gamma=self.config.gamma,
                    alpha_v=self.config.alpha_v,
                    alpha_h=self.config.alpha_h,
                    q_range=self.config.q_range,
                    log_num_actions=None,
                )
                grad_norm = self.net.update(loss)

                self.logger.debug(
                    "SR-DRL step %d: r=%.2f, v=%.2f, v_next=%.2f, "
                    "loss=%.4f, ent=%.4f, grad_norm=%.4f",
                    num_steps,
                    reward,
                    value,
                    next_value,
                    float(loss.item()),
                    float(entropy.item()),
                    grad_norm,
                )

            observation = next_obs
            last_value = value

        end_reason = None
        if observation.info:
            end_reason = observation.info.get("end_reason")

        self.logger.info(
            "SR-DRL episode done: return=%.2f, steps=%d, end_reason=%s",
            total_return,
            num_steps,
            str(end_reason),
        )
        return total_return, num_steps, end_reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SR-DRL inspired graph-based attacker agent for NetSecGame."
    )
    parser.add_argument(
        "--host",
        help="Host where the game server is",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        help="Port where the game server is",
        default=9000,
        type=int,
    )
    parser.add_argument(
        "--episodes",
        help="Number of episodes to run.",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--gamma",
        help="Discount factor gamma.",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--alpha_v",
        help="Value loss scaling.",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--alpha_h",
        help="Entropy regularisation scaling.",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate for the graph policy network.",
        default=3e-3,
        type=float,
    )
    parser.add_argument(
        "--mp_iterations",
        help="Number of message-passing iterations in the GNN.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--agent_seed",
        "--agent-seed",
        help="Random seed for the SR-DRL agent.",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--logdir",
        help="Directory to store logs.",
        default=path.join(
            path.dirname(path.abspath(__file__)),
            "logs",
        ),
    )
    parser.add_argument(
        "--trajectoriesdir",
        help="Directory to store trajectories as JSONL.",
        default=path.join(
            path.dirname(path.abspath(__file__)),
            "trajectories",
        ),
    )
    parser.add_argument(
        "--env_conf",
        help="Configuration file of the env. Only for logging purposes.",
        default="./env/netsecenv_conf.yaml",
        type=str,
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights & Biases logging (if installed).",
    )
    parser.add_argument(
        "--wandb_project",
        default="netsec-sr-drl",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="Wandb entity (team/user).",
    )
    parser.add_argument(
        "--wandb_group",
        default=None,
        help="Wandb group name.",
    )
    parser.add_argument(
        "--experiment_id",
        help="Id of the experiment to record into Wandb (optional).",
        default="",
        type=str,
    )
    parser.add_argument(
        "--models_dir",
        help="Directory to store/load SR-DRL models.",
        default=path.join(
            path.dirname(path.abspath(__file__)),
            "models",
        ),
        type=str,
    )
    parser.add_argument(
        "--previous_model",
        help="Path to a previous SR-DRL model checkpoint (.pt) to start from.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Run in evaluation mode (no learning).",
    )
    parser.add_argument(
        "--early_stop_winrate",
        help=(
            "In training mode, stop early when running win rate "
            "reaches this value. Set <= 0 to disable."
        ),
        default=0.95,
        type=float,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    if not path.exists(args.logdir):
        makedirs(args.logdir, exist_ok=True)
    if args.trajectoriesdir and not path.exists(args.trajectoriesdir):
        makedirs(args.trajectoriesdir, exist_ok=True)
    if args.models_dir and not path.exists(args.models_dir):
        makedirs(args.models_dir, exist_ok=True)
    logging.basicConfig(
        filename=path.join(args.logdir, "sr_drl_agent.log"),
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    config = SRDRLConfig(
        gamma=args.gamma,
        alpha_v=args.alpha_v,
        alpha_h=args.alpha_h,
        lr=args.lr,
        mp_iterations=args.mp_iterations,
        seed=args.agent_seed,
    )

    agent = SRDRLAttackerAgent(
        args.host,
        args.port,
        role=AgentRole.Attacker,
        config=config,
        logdir=args.logdir,
        capture_trajectories=bool(args.trajectoriesdir),
    )

    # Optionally load a previous model checkpoint
    if args.previous_model:
        try:
            agent.net.load(args.previous_model)
            logging.getLogger("SRDRLAttackerAgent").info(
                "Loaded previous model from %s", args.previous_model
            )
        except FileNotFoundError:
            logging.getLogger("SRDRLAttackerAgent").warning(
                "Previous model file not found: %s", args.previous_model
            )
        except Exception as e:
            logging.getLogger("SRDRLAttackerAgent").warning(
                "Could not load previous model %s: %s", args.previous_model, e
            )

    use_wandb = (wandb is not None) and (not args.disable_wandb)
    if use_wandb:
        run_name = None
        if args.experiment_id:
            run_name = f"SRDRLAgent.ID{args.experiment_id}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            config={
                "gamma": args.gamma,
                "alpha_v": args.alpha_v,
                "alpha_h": args.alpha_h,
                "lr": args.lr,
                "mp_iterations": args.mp_iterations,
                "episodes": args.episodes,
                "testing": args.testing,
                "agent_seed": args.agent_seed,
                "env_conf": args.env_conf,
                "experiment_id": args.experiment_id,
            },
        )
        # Try to save the environment configuration file, if it exists
        try:
            if path.exists(args.env_conf):
                wandb.save(
                    args.env_conf,
                    base_path=path.dirname(path.abspath(args.env_conf)),
                )
            else:
                logging.getLogger("SRDRLAttackerAgent").warning(
                    "Environment config file not found: %s", args.env_conf
                )
        except Exception as e:  # pragma: no cover - logging path
            logging.getLogger("SRDRLAttackerAgent").warning(
                "Could not save env config file to Wandb: %s", e
            )

    returns: list[float] = []
    steps: list[int] = []
    wins = 0
    detections = 0
    timeouts = 0

    for episode in range(1, args.episodes + 1):
        total_return, num_steps, end_reason = agent.play_episode(
            training=not args.testing
        )
        returns.append(total_return)
        steps.append(num_steps)

        if end_reason == AgentStatus.Success:
            wins += 1
        elif end_reason == AgentStatus.Fail:
            detections += 1
        elif end_reason == AgentStatus.TimeoutReached:
            timeouts += 1

        win_rate = wins / episode

        logging.getLogger("SRDRLAttackerAgent").info(
            "Episode %d/%d: return=%.2f, steps=%d, win_rate=%.3f, end_reason=%s",
            episode,
            args.episodes,
            total_return,
            num_steps,
            win_rate,
            str(end_reason),
        )
        if use_wandb:
            wandb.log(
                {
                    "episode": episode,
                    "return": total_return,
                    "steps": num_steps,
                    "avg_return": float(np.mean(returns)),
                    "win_rate": win_rate,
                    "wins": wins,
                    "detections": detections,
                    "timeouts": timeouts,
                },
                step=episode,
            )

        # Early stopping only in training mode
        if (not args.testing) and (args.early_stop_winrate > 0.0):
            if win_rate >= args.early_stop_winrate:
                logging.getLogger("SRDRLAttackerAgent").info(
                    "Early stopping at episode %d: win_rate=%.3f "
                    "(threshold=%.3f)",
                    episode,
                    win_rate,
                    args.early_stop_winrate,
                )
                break

    # After finishing episodes (including potential early stop), request one
    # more reset to capture the last episode's trajectory if enabled.
    if args.trajectoriesdir:
        try:
            agent.request_game_reset()
        except Exception as e:
            logging.getLogger("SRDRLAttackerAgent").warning(
                "Final request_game_reset() for trajectory capture failed: %s", e
            )

        if agent.trajectories:
            if args.testing:
                traj_file = "trajectories_testing.jsonl"
            else:
                traj_file = "trajectories_training.jsonl"
            traj_path = path.join(args.trajectoriesdir, traj_file)
            with open(traj_path, "w", encoding="utf-8") as f:
                f.writelines(agent.trajectories)
            logging.getLogger("SRDRLAttackerAgent").info(
                "Stored %d trajectories to %s",
                len(agent.trajectories),
                traj_path,
            )

    # Save final model checkpoint
    if args.models_dir:
        mode = "testing" if args.testing else "training"
        ckpt_name = f"sr_drl_agent.{mode}"
        if args.experiment_id:
            ckpt_name += f".experiment{args.experiment_id}"
        ckpt_name += ".pt"
        ckpt_path = path.join(args.models_dir, ckpt_name)
        try:
            agent.net.save(ckpt_path)
            logging.getLogger("SRDRLAttackerAgent").info(
                "Saved SR-DRL model checkpoint to %s", ckpt_path
            )
        except Exception as e:
            logging.getLogger("SRDRLAttackerAgent").warning(
                "Failed to save SR-DRL model checkpoint to %s: %s",
                ckpt_path,
                e,
            )

    # Terminate connection gracefully after all episodes
    agent.terminate_connection()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
