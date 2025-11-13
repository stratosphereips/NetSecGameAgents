# ------------------------------------------------------------
#   Title: maml_agent.py
#   Purpose: To train a single maml agent with multiple tasks (env) 
#   Author: Jihoon Shin - jshin4@miners.utep.edu
# ------------------------------------------------------------ 
import sys
import os
import logging
import mlflow
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime, timezone
import socket
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from os import path, makedirs
from random import choice
from netaddr import IPNetwork
from netaddr import IPAddress as IP
from collections import deque, defaultdict
from AIDojoCoordinator.game_components import Action, ActionType, Observation, AgentStatus, GameState
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions
from NetSecGameAgents.agents.attackers.random.random_agent import RandomAttackerAgent
from NetSecGameAgents.agents.attackers.q_learning.q_agent import QAgent
import psutil
import csv
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save output to both terminal and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()


log_filename = f"maml_agent_log.txt"
sys.stdout = Logger(log_filename)

# JSONL trajectory of evaluation episodes logger
class TrajectoryLogger:
    """
    Writes one JSONL file per episode into base_dir and,
    on close, also writes a human-readable TXT into base_dir/readable/.
    """
    def __init__(self, base_dir, run_id):
        self.base_dir = base_dir
        self.readable_dir = path.join(base_dir, "readable")
        makedirs(self.base_dir, exist_ok=True)
        makedirs(self.readable_dir, exist_ok=True)

        self.run_id = run_id
        self.fp = None
        self._records = []
        self._current_jsonl_path = ""
        self._current_txt_path = ""

    def write_manifest(self, manifest: dict):
        mpath = path.join(self.base_dir, "manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def open_episode(self, epoch: int, episode_id: int = 0):
        base = f"tmp_epoch_{epoch:04d}__episode_{episode_id:04d}"
        self._current_jsonl_path = path.join(self.base_dir, base + ".jsonl")
        self._current_txt_path   = path.join(self.readable_dir, base + "_readable.txt")
        # reset for this episode
        self._records = []
        self.fp = open(self._current_jsonl_path, "w", encoding="utf-8")
        return self._current_jsonl_path

    def log_step(self, row: dict):
        if self.fp is None:
            raise RuntimeError("TrajectoryLogger: call open_episode() first.")
        row.setdefault("schema_version", "v1")
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fp.flush()
        self._records.append(row)

    def close_episode(self):
        if self.fp:
            self.fp.close()
            self.fp = None
        if self._records:
            txt = self._to_readable_text(self._records)
            with open(self._current_txt_path, "w", encoding="utf-8") as f:
                f.write(txt)
        self._records = []
        self._current_jsonl_path = ""
        self._current_txt_path = ""

    # ---------- Formatting helpers ----------
    @staticmethod
    def _fmt_snapshot(snap: dict) -> str:
        if not isinstance(snap, dict): snap = {}
        nets = ", ".join(snap.get("known_networks", [])) or "-"
        return (
            f"Known networks : {nets}\n"
            f"Known hosts    : {snap.get('num_known_hosts', 0)}\n"
            f"Controlled     : {snap.get('num_controlled_hosts', 0)}\n"
            f"Known services : {snap.get('num_known_services', 0)}\n"
            f"Known data     : {snap.get('num_known_data_items', 0)}"
        )

    @staticmethod
    def _fmt_group_line(name: str, group: dict) -> str:
        if not group: return f"{name:>7}: -"
        kv = ", ".join(f"{k}={v}" for k, v in group.items())
        return f"{name:>7}: {kv}"

    def _fmt_header(self, rec: dict) -> str:
        goal = rec.get("goal") or {}
        snap = rec.get("initial_snapshot") or {}
        return (
            "=== EPISODE SUMMARY ===\n"
            f"Run ID         : {rec.get('run_id')}\n"
            f"Trajectory     : {rec.get('trajectory_id')}\n"
            f"Epoch/Episode  : {rec.get('epoch')}/{rec.get('episode_id')}\n"
            f"Phase          : {rec.get('phase')}\n"
            f"Timestamp      : {rec.get('timestamp')}\n"
            f"Goal           : {goal.get('desc')}\n"
            "\n-- Initial Snapshot --\n"
            f"{self._fmt_snapshot(snap)}\n"
        )

    def _fmt_step(self, rec: dict) -> str:
        sa = rec.get("selected_action") or {}
        st = rec.get("state_named") or {}
        source = st.get("source") or {}
        target = st.get("target") or {}
        globl  = st.get("global") or {}

        parts = [
            f"\n=== STEP {rec.get('step')} ===  @ {rec.get('timestamp')}",
            "-- State --",
        ]

        obs_text = rec.get("obs_text")
        if obs_text:
            parts.append(obs_text)

        parts.extend([
            self._fmt_group_line("source", source),
            self._fmt_group_line("target", target),
            self._fmt_group_line("global", globl),
            "", 
            f"Action         : {sa.get('action_type')}",
            f"From/To        : {sa.get('source_host')} -> {sa.get('target_host')}",
            f"Params         : {sa.get('params')}",
            f"Chosen prob    : {rec.get('chosen_prob')}",
            f"Reward         : {rec.get('reward_before')} -> {rec.get('reward_after')}",
            f"Done next      : {rec.get('done_next')}",
        ])
        return "\n".join(parts)

    @staticmethod
    def _fmt_footer(rec: dict) -> str:
        return (
            "\n=== EPISODE END ===\n"
            f"Steps total    : {rec.get('steps')}\n"
            f"Episode return : {rec.get('return')}\n"
            f"End reason     : {rec.get('end_reason')}\n"
            f"Timestamp      : {rec.get('timestamp')}\n"
        )

    def _to_readable_text(self, records: list[dict]) -> str:
        header  = next((r for r in records if r.get("record_type") == "trajectory_header"), None)
        footer  = next((r for r in records if r.get("record_type") == "trajectory_footer"), None)
        steps   = [r for r in records if r.get("step") is not None]
        steps.sort(key=lambda r: r.get("step", 0))

        lines = []
        if header: lines.append(self._fmt_header(header))
        if steps:
            lines.append("\n=== STEPS ===")
            for s in steps:
                lines.append(self._fmt_step(s))
        if footer: lines.append(self._fmt_footer(footer))
        return "\n".join(lines).rstrip() + "\n"


class PolicyNetwork(nn.Module):
    # Input: observations (state: 12D vector (source:3, target:6, global:3) )
    # Output: probability distribution over 5 actions (ScanNetwork, FindServices, FindData, ExploitService, ExfiltrateData)
    def __init__(self, input_dim=12, hidden_dim=(64,64), output_dim=5):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            logits = self.fc3(x)

        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            logits = F.linear(x, params['fc3.weight'], params['fc3.bias'])

        return logits


# Model Agnostic Meta Learning agent
class MAMLAgent(BaseAgent):

    def __init__(self, host, port, role, policy, inner_lr, outer_lr, inner_steps, max_steps, num_meta_batches, num_eval_batches, meta_batch_size, test_batch_size, eval_meta_batch_size, eval_test_batch_size, request_trajectory):
        super().__init__(host, port, role)
        self.policy = policy
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.outer_lr)
        self.max_steps = max_steps 
        self.num_meta_batches = num_meta_batches    # [meta_batch_size, test_batch_size] * num_meta_batches
        self.num_eval_batches = num_eval_batches    # for evaluation
        self.meta_batch_size = meta_batch_size
        self.test_batch_size = test_batch_size
        self.eval_meta_batch_size = eval_meta_batch_size
        self.eval_test_batch_size = eval_test_batch_size
        self.gamma = 0.99  # Discount factor
        self.request_trajectory = request_trajectory
    
    # feature specification for manifest 
    FEATURE_SPEC = {
        "feature_version": "fs_v1",
        "vector_order": [
            {"name":"source.num_services_on_source",     "index":0,  "dtype":"int",   "norm":"identity"},
            {"name":"source.in_known_network",           "index":1,  "dtype":"bin",   "norm":"identity"},
            {"name":"source.reachable_ratio",            "index":2,  "dtype":"float", "norm":"[0,1]"},
            {"name":"target.stage_progress",             "index":3,  "dtype":"float", "norm":"stage/4"},
            {"name":"target.is_compromised",             "index":4,  "dtype":"bin",   "norm":"identity"},
            {"name":"target.num_services_on_target",     "index":5,  "dtype":"int",   "norm":"identity"},
            {"name":"target.has_known_data",             "index":6,  "dtype":"bin",   "norm":"identity"},
            {"name":"target.is_blocked_from_source",     "index":7,  "dtype":"bin",   "norm":"identity"},
            {"name":"target.degree_in_discovery_graph",  "index":8,  "dtype":"int",   "norm":"identity"},
            {"name":"global.owned_ratio",                "index":9,  "dtype":"float", "norm":"[0,1]"},
            {"name":"global.avg_degree",                 "index":10, "dtype":"float", "norm":"identity"},
            {"name":"global.graph_density",              "index":11, "dtype":"float", "norm":"[0,1]"}
        ],
        "groups": {"source":[0,1,2], "target":[3,4,5,6,7,8], "global":[9,10,11]},
    }
    # turn a flat vector into named groups using FEATURE_SPEC
    def vector_to_named(self, vec):
        named = {"source": {}, "target": {}, "global": {}}
        # vec can be tensor or list
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().tolist()
        for item in self.FEATURE_SPEC["vector_order"]:
            idx = item["index"]
            if idx < len(vec):
                group, key = item["name"].split(".", 1)
                named[group][key] = float(vec[idx])
        return named

    # Source Node Features: currently controlling node
    def get_source_features(self, state: GameState, source: IP) -> list:
        num_services_discovered_on_source = len(state.known_services.get(source, [])) # How much we know about this node. 
        
        networks = [IPNetwork(str(net)) for net in state.known_networks]
        source_in_known_network = int(any(str(source) in net for net in networks))  # Indicates whether the agent knows the subnet this source belongs to
        
        # How “central” the current source is in the known graph.
        _, _, graph_edges, index_to_node = state.as_graph
        node_to_index = {node: idx for idx, node in index_to_node.items()}
        source_idx = node_to_index.get(source)  # to look up the graph index for source IP 

        if source_idx is None:
            reachable_ratio = 0
        else: # Breadth-First Search to count how many discovered nodes are reachable from the current source node
            # build adjacency list
            # adjacency = {i: [] for i in range(len(index_to_node))}
            # for u, v in graph_edges:
            #     adjacency[u].append(v)

            adjacency = defaultdict(list)
            for u, v in graph_edges:
                adjacency[u].append(v)


            # Breadth-First Search from source_idx
            visited = set()
            queue = deque([source_idx])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    queue.extend(adjacency[current])
            reachable_ratio = len(visited) / (len(state.known_hosts) + 1e-5)

        source_features = [
            num_services_discovered_on_source,
            source_in_known_network,
            reachable_ratio     
        ]
        return source_features
    # --------------------------------------------------------------------------------------------------
    # Target Node Features: A previously discovered node the agent is trying to access/own/exploit
    def get_target_features(self, state: GameState, target: IP, source: IP) -> list:
        is_compromised = int(target in state.controlled_hosts)  # Whether the agent already owns this node
        num_services_discovered_on_target = len(state.known_services.get(target, [])) # How much we know about this node's services
        has_known_data = int(target in state.known_data and len(state.known_data[target]) > 0)  # Whether the agent has discovered data on this node
        is_blocked_from_source = int(source in state.known_blocks.get(target, set())) # Whether the target is blocked from the current source
        
        # How connected the target is in the partial known topology
        _, _, graph_edges, index_to_node = state.as_graph
        node_to_index = {node: idx for idx, node in index_to_node.items()} 
        target_idx = node_to_index.get(target) # to look up the graph index for target IP 
        if target_idx is None:
            degree_in_discovery_graph = 0
        else:  # Count how many edges involve this target node
            degree_in_discovery_graph = sum(1 for u, v in graph_edges if target_idx in (u, v)) 
        
        # Agent's stage of target node
        scanned      = int(target in state.known_hosts)          # ScanNetwork done?
        services     = int(len(state.known_services.get(target, [])) > 0)  # FindServices done? 0 or 1, no matter how many services
        exploited    = int(target in state.controlled_hosts)     # ExploitService done?
        data_found   = int(len(state.known_data.get(target, [])) > 0) # 0 or 1
        # now build an ordered stage 
        if   not scanned:            stage = 0
        elif not services:           stage = 1
        elif not exploited:          stage = 2
        elif not data_found:         stage = 3
        else:                        stage = 4
        # stage_one_hot = [int(stage == i) for i in range(5)]   # Let's do this if policy don't works.

        target_features = [
            stage/4, 
            is_compromised,
            num_services_discovered_on_target,
            has_known_data,
            is_blocked_from_source,
            degree_in_discovery_graph
        ]

        return target_features
    # --------------------------------------------------------------------------------------------------
    # Global Features: From the environment and agent's view
    def get_global_features(self, state: GameState, source: IP) -> list:
        
        # Progress: How much progress the agent has made in compromising discovered nodes.
        num_owned_nodes = len(state.controlled_hosts)
        num_discovered_nodes = len(state.known_hosts)
        owned_ratio = num_owned_nodes / (num_discovered_nodes + 1e-5) 

        # Get discovery graph info
        _, _, graph_edges, index_to_node = state.as_graph
        num_nodes = len(index_to_node)
        num_edges = len(graph_edges) // 2  # Edges are bidirectional 

        # How connected the discovered network is. (Average number of edges per node)
        avg_degree = (2 * num_edges) / (num_nodes + 1e-5)

        # Edge density normalized over all possible connections.
        graph_density = (2 * num_edges) / (num_nodes * (num_nodes - 1) + 1e-5)

        global_features = [
            owned_ratio,
            avg_degree,
            graph_density
        ]
        return global_features
    # -------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------
    # Full State Vector Builder for MAML Policy
    def build_state_vector(self, state: GameState, source: IP, target: IP, action) -> list:
        """
        Concatenates source, target, and global features into a single flat state vector.
        This is used as input to the policy network.
        """
        source_features = self.get_source_features(state, source)
        if action.action_type == ActionType.ExfiltrateData: 
            # Change src and tgt since we are moving data from src to tgt.
            target_features = self.get_target_features(state, source, target) 
        else:     
            target_features = self.get_target_features(state, target, source)
        global_features = self.get_global_features(state, source)

        full_state_vector = source_features + target_features + global_features
        return full_state_vector


    def idx_to_actiontype(self, index: int) -> ActionType:
        action_type_list = [
            ActionType.ScanNetwork,
            ActionType.FindServices,
            ActionType.FindData,
            ActionType.ExploitService,
            ActionType.ExfiltrateData
        ]
        return action_type_list[index]

    def select_action(self, observation: Observation, params=None):
        state = observation.state
        # self._logger.info("[Current Observation]")
        # self._logger.info(f"  • Known Networks : {', '.join(map(str, state.known_networks))}")
        # self._logger.info(f"  • Known Hosts    : {', '.join(map(str, state.known_hosts))}")
        # self._logger.info(f"  • Controlled     : {', '.join(map(str, state.controlled_hosts))}")
        # self._logger.info(f"  • Known Services : { {str(k): [s.name for s in v] for k, v in state.known_services.items()} }")
        # self._logger.info("  • Known Data     :")
        # for host_ip, data_items in state.known_data.items():
        #     self._logger.info(f"     - Host {host_ip}:")
        #     for d in data_items:
        #         self._logger.info(f"         • ID: {d.id}, Owner: {d.owner}, Type: {d.type}, Size: {d.size}")
        # self._logger.info(f"  • Blocked Hosts  : { {str(k): [str(b) for b in v] for k, v in state.known_blocks.items()} }\n")
        
        valid_actions = generate_valid_actions(state)
        # self._logger.info(f"        - valid_actions: {valid_actions}")

        if not valid_actions:
            return None, None, None, None

        # Build ActionType→idx mapping via idx_to_actiontype
        type2idx = { self.idx_to_actiontype(i): i for i in range(5) }

        batch_vecs = []
        mapping   = []  # (action_obj, its_index)

        for a in valid_actions:
            src = a.parameters.get("source_host")
            if src is None:
                continue
            tgt = a.parameters.get("target_host")  # may be None

            # Build feature‐vector
            if tgt is None:
                src_feats    = self.get_source_features(state, src)
                global_feats = self.get_global_features(state, src)
                pad_len = len(self.get_target_features(state, src, src))
                tgt_feats    = [0.0] * pad_len
                vec = src_feats + tgt_feats + global_feats
            else:
                vec = self.build_state_vector(state, src, tgt, a)

            # print(f"    - feature‐vector: {vec}")
            batch_vecs.append(torch.tensor(vec, dtype=torch.float32))
            mapping.append((a, type2idx[a.action_type]))

        if not batch_vecs:
            return None, None, None, None

        batch     = torch.stack(batch_vecs).to(device)       # [N,D]
        logits    = self.policy(batch, params=params)       # [N,5]
        # [3rd approach] 
        # it learn “increase the probability of the chosen candidate among all candidates,” not just “like this action type.""
        # Build a tensor of action-type indices (one per candidate)
        type_idx_tensor = torch.tensor(
            [idx for (_, idx) in mapping], device=logits.device, dtype=torch.long
        )  # [N]

        # Candidate logits: pick the logit for each candidate's ActionType → [N]
        cand_logits = logits.gather(dim=1, index=type_idx_tensor.view(-1, 1)).squeeze(1)  # [N]

        # Make a single N-way distribution over candidates and sample
        dist = Categorical(logits=cand_logits)  # Converts bigger logits into higher probabilities
        best = dist.sample().item()  # index in [0..N-1]

        best_action, _ = mapping[best]
        best_vec = batch_vecs[best]

        return best_action, best_vec, cand_logits, best

    def play_game(self, params=None, randomize_topology=False, 
                    traj_logger: 'TrajectoryLogger' = None, traj_meta: dict = None, request_trajectory: bool = False):
        """
        Reset → play one full episode → return the terminal observation + stats.
        If traj_logger is provided, logs a step-by-step trajectory in JSONL.
        """
        observation = self.request_game_reset(
            request_trajectory=request_trajectory,
            randomize_topology=randomize_topology
            )  
        
        last_observation = observation
        total_reward = 0
        training_data = []
        num_steps = 0
        episodic_returns = []

        goal_info = observation.info if observation and observation.info else {}
        goal_ip = goal_info.get("goal_ip", None) or goal_info.get("goal", None)
        goal_desc = goal_info.get("goal_description", "Unknown")

        # self._logger.info("Playing one episode")
        # self._logger.info(f"    → Goal: {goal_desc}")

        # compute trajectory identifiers once and write a header row
        epoch_for_id = (traj_meta or {}).get("epoch", -1)
        trajectory_id = f"epoch-{int(epoch_for_id):04d}" if isinstance(epoch_for_id, (int, float)) else str(epoch_for_id)

        if traj_logger is not None:
            st = observation.state
            init_snapshot = {
                "known_networks": [str(n) for n in st.known_networks],
                "num_known_hosts": len(st.known_hosts),
                "num_controlled_hosts": len(st.controlled_hosts),
                "num_known_services": sum(len(v) for v in st.known_services.values()),
                "num_known_data_items": sum(len(v) for v in st.known_data.values()),
            }
            traj_logger.log_step({
                "schema_version": "v1",
                "record_type": "trajectory_header",
                "run_id": getattr(traj_logger, "run_id", None),
                "trajectory_id": trajectory_id,
                "epoch": epoch_for_id,
                "phase": (traj_meta or {}).get("phase", "eval_query"),
                "episode_id": (traj_meta or {}).get("episode_id", 0),
                "goal": {"ip": str(goal_ip) if goal_ip else None, "desc": goal_desc},
                "initial_snapshot": init_snapshot,
            })
        
        # 2) roll out until terminal
        # TODO What if it ends sooner ???
        for step in range(self.max_steps):
            num_steps += 1
            action, state_vec, logits, action_idx = self.select_action(observation, params)
            obs_text_str = self._observation_block_text(observation.state)

            prev_reward = float(last_observation.reward or 0.0)
            new_obs = self.make_step(action)
            next_reward = float(new_obs.reward or 0.0)

            step_reward = next_reward

            if state_vec is not None:
                training_data.append((state_vec, logits, action_idx, step_reward))
                nn_input = state_vec.detach().cpu().tolist()
                state_named = self.vector_to_named(nn_input)
            else:
                nn_input = []
                state_named = {}
            
            # accumulate returns
            episodic_returns.append(step_reward)
            total_reward += step_reward
            
            # log trajectory 
            if traj_logger is not None and action is not None:
                chosen_logit = float(logits[action_idx].item()) if logits is not None else None
                chosen_prob  = float(F.softmax(logits, dim=0)[action_idx].item()) if logits is not None else None

                src = action.parameters.get("source_host")
                tgt = action.parameters.get("target_host")
                goal_info = new_obs.info if new_obs and new_obs.info else {}
                goal_ip = goal_info.get("goal_ip", None) or goal_info.get("goal", None)
                goal_desc = goal_info.get("goal_description", "Unknown")
                next_end = bool(new_obs.end)

                row = {
                    "schema_version": "v1",
                    "run_id": getattr(traj_logger, "run_id", None),
                    "trajectory_id": trajectory_id,                   # <— epoch-based
                    "epoch": epoch_for_id,
                    "phase": (traj_meta or {}).get("phase", "eval_query"),
                    "episode_id": (traj_meta or {}).get("episode_id", 0),
                    "step": num_steps,
                    "goal": {"ip": str(goal_ip) if goal_ip else None, "desc": goal_desc},
                    "state_named": state_named,
                    "nn_input": nn_input,
                    "selected_action": {
                        "action_type": action.action_type.name,
                        "source_host": str(src) if src is not None else None,
                        "target_host": str(tgt) if tgt is not None else None,
                        "params": {k: str(v) for k, v in (action.parameters or {}).items()}
                    },
                    "chosen_logit": chosen_logit,
                    "chosen_prob": chosen_prob,
                    "reward_before": prev_reward,
                    "reward_after": next_reward,
                    "done_next": next_end,
                    "obs_text": obs_text_str,
                }
                traj_logger.log_step(row)
            
            observation = new_obs
            last_observation = new_obs

            if last_observation.end:
                break
           
        # 3) at end, last_observation holds our final valid step
        total_reward = np.sum(episodic_returns)
        # self._logger.info(f"Episode ended → return={total_reward}, steps={num_steps}")
        # self._logger.info(f"Final Observation → end={last_observation.end}, info={last_observation.info}")

        if traj_logger is not None:
            info = last_observation.info or {}
            end_reason = info.get("end_reason")
            if hasattr(end_reason, "name"):
                end_reason = end_reason.name
            traj_logger.log_step({
                "schema_version": "v1",
                "record_type": "trajectory_footer",
                "run_id": getattr(traj_logger, "run_id", None),
                "trajectory_id": trajectory_id,
                "epoch": epoch_for_id,
                "phase": (traj_meta or {}).get("phase", "eval_query"),
                "episode_id": (traj_meta or {}).get("episode_id", 0),
                "steps": num_steps,
                "return": float(total_reward),
                "end_reason": end_reason,
            })

        return last_observation, total_reward, num_steps, training_data
    
    def collect_episodes(self, total_episodes: int, params=None, randomize_topology=False, 
                     traj_logger: 'TrajectoryLogger' = None, traj_meta: dict = None,
                     log_one_idx: int | None = None, log_all: bool = False):
        episodes = []
        training_data_all = []
        all_total_rewards = []
        wins = detected = max_steps = 0
        num_win_steps, num_detected_steps, num_max_steps_steps = [], [], []
        num_win_returns, num_detected_returns, num_max_steps_returns = [], [], []

        # register once
        # self.register()

        for i in range(total_episodes):
            
            # only log ONE chosen episode (e.g., i == 0 for "first")
            # use_logger = traj_logger if (traj_logger is not None and log_one_idx is not None and i == log_one_idx) else None
            
            use_logger = None
            if traj_logger is not None:
                if log_all:
                    use_logger = traj_logger
                elif log_one_idx is not None and i == log_one_idx:
                    use_logger = traj_logger
            
            if use_logger is not None:
                use_logger.open_episode(epoch=(traj_meta or {}).get("epoch", -1), episode_id=i)
            
            try: 
                meta = dict(traj_meta or {})
                meta["episode_id"] = i
                last_obs, total_reward, num_steps, training_data = self.play_game(
                    params,
                    randomize_topology=randomize_topology,
                    traj_logger=use_logger,
                    traj_meta=meta if use_logger is not None else None,
                    request_trajectory=self.request_trajectory
                )
                # print(f"[Episode {i+1}] Steps: {num_steps:>3}, Total Reward: {total_reward:>5.1f}, End: {last_obs.info.get('end_reason') if last_obs.info else 'N/A'}")
            finally:
                if use_logger is not None:
                    use_logger.close_episode()

            info = last_obs.info
            if info and info.get('end_reason') == AgentStatus.Fail:
                detected += 1
                num_detected_steps.append(num_steps)
                num_detected_returns.append(total_reward)
            elif info and info.get('end_reason') == AgentStatus.Success:
                wins += 1
                num_win_steps.append(num_steps)
                num_win_returns.append(total_reward)
            elif info and info.get('end_reason') == AgentStatus.TimeoutReached:
                max_steps += 1
                num_max_steps_steps.append(num_steps)
                num_max_steps_returns.append(total_reward)
                

            episodes.append(last_obs)
            training_data_all.append(training_data)
            all_total_rewards.append(total_reward)

        self.log_performance(
            episode=total_episodes,
            wins=wins, detected=detected, max_steps=max_steps,
            num_win_steps=num_win_steps, num_detected_steps=num_detected_steps, num_max_steps_steps=num_max_steps_steps,
            num_win_returns=num_win_returns, num_detected_returns=num_detected_returns, num_max_steps_returns=num_max_steps_returns,
            epoch=(traj_meta or {}).get("epoch")  
        )
        win_rate = (wins / total_episodes) * 100
        return episodes, training_data_all, all_total_rewards, win_rate
    
    def _observation_block_text(self, state) -> str:
        lines = []
        lines.append("[Current Observation]")
        lines.append(f"  • Known Networks : {', '.join(map(str, state.known_networks))}")
        lines.append(f"  • Known Hosts    : {', '.join(map(str, state.known_hosts))}")
        lines.append(f"  • Controlled     : {', '.join(map(str, state.controlled_hosts))}")
        lines.append(f"  • Known Services : { {str(k): [getattr(s,'name',str(s)) for s in v] for k, v in state.known_services.items()} }")
        lines.append("  • Known Data     :")
        for host_ip, data_items in state.known_data.items():
            lines.append(f"     - Host {host_ip}:")
            for d in data_items:
                lines.append(f"         • ID: {getattr(d,'id','?')}, Owner: {getattr(d,'owner','?')}, "
                            f"Type: {getattr(d,'type','?')}, Size: {getattr(d,'size','?')}")
        lines.append(f"  • Blocked Hosts  : { {str(k): [str(b) for b in v] for k, v in state.known_blocks.items()} }")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    def log_performance(self, episode, wins, detected, max_steps,
                    num_win_steps, num_detected_steps, num_max_steps_steps,
                    num_win_returns, num_detected_returns, num_max_steps_returns,
                    epoch=None):

        eval_win_rate = (wins / episode) * 100
        eval_detection_rate = (detected / episode) * 100
        eval_average_returns = np.mean(num_detected_returns + num_win_returns + num_max_steps_returns)
        eval_std_returns = np.std(num_detected_returns + num_win_returns + num_max_steps_returns)
        eval_average_episode_steps = np.mean(num_win_steps + num_detected_steps + num_max_steps_steps)
        eval_std_episode_steps = np.std(num_win_steps + num_detected_steps + num_max_steps_steps)
        eval_average_win_steps = np.mean(num_win_steps) if num_win_steps else 0
        eval_std_win_steps = np.std(num_win_steps) if num_win_steps else 0
        eval_average_detected_steps = np.mean(num_detected_steps) if num_detected_steps else 0
        eval_std_detected_steps = np.std(num_detected_steps) if num_detected_steps else 0
        eval_average_max_steps_steps = np.mean(num_max_steps_steps) if num_max_steps_steps else 0
        eval_std_max_steps_steps = np.std(num_max_steps_steps) if num_max_steps_steps else 0

        text = f'''
        Final evaluation after {episode} episodes:
            Wins={wins}, Detections={detected},
            Win rate={eval_win_rate:.2f}%,
            Detection rate={eval_detection_rate:.2f}%,
            Avg return={eval_average_returns:.2f} ± {eval_std_returns:.2f},
            Avg episode steps={eval_average_episode_steps:.2f} ± {eval_std_episode_steps:.2f},
            Avg win steps={eval_average_win_steps:.2f} ± {eval_std_win_steps:.2f},
            Avg detected steps={eval_average_detected_steps:.2f} ± {eval_std_detected_steps:.2f},
            Avg timeout steps={eval_average_max_steps_steps:.2f} ± {eval_std_max_steps_steps:.2f}
        '''
        # print(text)
        self._logger.info(text)

        # --- Append compact CSV for the latest evaluation batch (for report the results) ---
        try:
            csv_path = getattr(self, "_results_csv_path", None)
            if csv_path and (epoch is not None):  # only log during evaluation
                
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                header = ["epoch", "# Evaluation Episodes", "winrate", "average_returns",
                        "average_episode_steps", "detection_rate", "average_detection_steps"]
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(header)
                    w.writerow([
                        epoch,
                        episode,
                        f"{(eval_win_rate/100.0):.4f}",           # fraction (e.g., 0.2238)
                        f"{eval_average_returns:.3f}",
                        f"{eval_average_episode_steps:.3f}",
                        f"{(eval_detection_rate/100.0):.4f}",     # fraction
                        f"{eval_average_detected_steps:.3f}",
                    ])
        except Exception as e:
            self._logger.info(f"[results_csv] failed to write: {e}")

    def process_training_data(self, training_data_all:list[list[tuple]])-> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts raw training data into log_probs and returns for policy gradient.
        Args:
            training_data_all: list of episodes; each ep is a list of
                (state_vec, cand_logits[N], chosen_idx, reward)
        Returns:
            log_probs [total_steps], returns [total_steps], []
        """
        per_ep = []
        for ep in training_data_all:
            if not ep:
                continue
            _, cand_logits_list, chosen_idx_list, rewards = zip(*ep)

            ep_log_probs = torch.stack([
                F.log_softmax(cand_logits, dim=0)[idx]
                for cand_logits, idx in zip(cand_logits_list, chosen_idx_list)
            ]).to(device)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            ep_returns = self.compute_returns(rewards_t)  # discounted within this episode
            per_ep.append((ep_log_probs, ep_returns))
        return per_ep

    def compute_returns(self, rewards:torch.Tensor)-> torch.Tensor:
        """
        Computes discounted returns for a single episode.
        Args:
            rewards: tensor of shape [T] with rewards for each time step
        Returns:
            returns: tensor of shape [T] with discounted returns
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=device)

    def compute_policy_loss_vanilla_RL(self, per_episode):
        """
        Computes policy loss using REINFORCE for multiple episodes.
        'per_episode' is a list of (log_probs_tensor, returns_tensor) for each episode.
        We normalize returns per episode and compute the loss.
        Args:
            per_episode: list of tuples (log_probs, returns) for each episode
        Returns:
            mean_loss: scalar tensor, average policy loss across episodes
        """
        if not per_episode:
            return torch.tensor(0.0, device=device)
        losses = []
        for log_probs, returns in per_episode:
            adv = (returns - returns.mean()) / (returns.std() + 1e-8)  # per-episode
            loss = -(log_probs * adv).sum()
            losses.append(loss)
            # print(f"[LOSS] returns={returns}, advantage= {adv}, loss= {loss}")
        return torch.stack(losses).mean()

    def inner_loop(self, flag_eval:bool, batch_size:int)-> tuple[dict, list]:
        """
        Performs inner loop adaptation for one task. The inner loop consists of self.inner_steps updates.
        Each update uses batch_size episodes to compute policy gradients and update the adapted parameters.
        Returns:
            adapted_params: dict of adapted parameters after inner loop
            before_returns: list of returns before adaptation
        Args:
            flag_eval: bool, whether in evaluation mode (affects batch size)
            batch_size: int, number of episodes to collect per 1 update inside the inner loop
        """
        if flag_eval: # evaluation
            batch_size = self.eval_meta_batch_size
        else: # training
            batch_size = self.meta_batch_size

        # Initialize adapted parameters as a copy of the initial policy parameters
        adapted_params = {name: param.clone().requires_grad_(True) for name, param in self.policy.named_parameters()}

        before_returns = []
        # after_returns = []

        # TODO MOVE THIS OUT OF inner_loop???
        # In this first episode, we should change the network layout by setting randomize_topology=True.
        # [For log] Before adaptation
        _, _, total_rewards, win_rate = self.collect_episodes(1, params=adapted_params, randomize_topology=True)
        before_returns.append(np.mean(total_rewards))
        

        # Inner loop adaptation steps
        # Play self.inner_steps episodes, each time updating adapted_params with policy gradient
        for i in range(self.inner_steps):
            # Collect batch size episodes using the current adapted parameters
            episodes, training_data_all, all_total_rewards, win_rate = self.collect_episodes(batch_size, params=adapted_params, randomize_topology=False) 
            # process training data per episode
            per_episode = self.process_training_data(training_data_all)

            # compute loss
            loss = self.compute_policy_loss_vanilla_RL(per_episode)
            # Compute gradients of the loss w.r.t. the adapted parameters
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, allow_unused=True)
            # Update adapted parameters using gradient descent
            new_adapted = {}
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is None:
                    new_adapted[name] = param  # skip update
                else:
                    new_adapted[name] = (param - self.inner_lr * grad).to(device)
            adapted_params = new_adapted
        # return adapted parameters - END OF inner loop
        # TODO: what is 'before returns' and 'after_returns' here????
        return adapted_params, before_returns

    def outer_loop(self)-> tuple[float, np.ndarray, float]:
        """
        Performs one outer loop meta-update across self.num_meta_batches tasks.
        Returns:
            meta_loss: float, the meta loss value
            avg_before: np.ndarray, average returns before adaptation across tasks
            avg_after: float, average returns after adaptation across tasks
        """
        # prepare accumulators 
        N = self.num_meta_batches
        B = self.inner_steps + 1
        acc_before = 0.0
        acc_after  = 0.0
        all_task_losses = [] 
        after_returns = []
        
        # collect meta-batch of tasks
        # each task: inner loop adaptation (self.inner_steps updates)
        # each tasks returns its adapted parameters + average loss after adaptation
        for i in range(self.num_meta_batches):
            # TODO - run topology change here???
            # run inner loop adaptation
            flag_eval = False
            adapted_params, before_returns = self.inner_loop(flag_eval=False, batch_size=self.meta_batch_size)
            # Query set evaluation (no further update) - evaluation for this task
            _, training_data_all, total_rewards, win_rate = self.collect_episodes(self.test_batch_size, params=adapted_params, randomize_topology=False)
            
            # accumulate returns across meta-batch
            acc_before += float(np.mean(before_returns))
            acc_after  += np.mean(total_rewards)

            per_episode = self.process_training_data(training_data_all)
            loss = self.compute_policy_loss_vanilla_RL(per_episode)

            all_task_losses.append(loss)
        
        # Meta-update: average loss across tasks
        self.optimizer.zero_grad()
        meta_loss = torch.stack(all_task_losses).mean()

        # ---- Regularization Block ----
        lmbda = 1e-4  # regularization weight
        l2_reg = torch.tensor(0., device=device)
        for param in self.policy.parameters():
            l2_reg += torch.norm(param, p=2)  # or p=1 for L1
        meta_loss += lmbda * l2_reg
        # -----------------------------
        # compute gradients and update initial policy parameters
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # now compute *average* before/after returns
        # avg_before = acc_before / N      # shape (inner_steps+1,)
        avg_before = np.array([acc_before / N], dtype=np.float32)  # single pre-adapt metric
        avg_after  = acc_after  / N      # scalar

        return meta_loss.item(), avg_before, avg_after

    def evaluate(self, traj_logger=None, epoch=None, log_one_idx=None):
        N = self.num_eval_batches
        pre_adapt_rewards = []
        after_win_rates = []
        after_rewards = []

        for i in range(self.num_eval_batches): 
            # print(f"\n[Eval Task {i+1}/{self.num_meta_batches}]")

            # inner loop adaptation
            # adapted_params, before_returns, after_returns = self.inner_loop()
            flag_eval = True
            adapted_params, before_returns = self.inner_loop(flag_eval)
            pre_adapt_rewards.append(np.mean(before_returns))

            # log only on the FIRST task (task_i == 0), ONE episode picked by log_one_idx
            use_logger = traj_logger if (traj_logger is not None and epoch is not None and i == 0 and log_one_idx is not None) else None
            
            # Query set evaluation (no further update)
            _, _, total_rewards, win_rate = self.collect_episodes(
                self.eval_test_batch_size,
                params=adapted_params,
                randomize_topology=False,
                traj_logger=traj_logger,                       # ← attach logger here
                traj_meta={"epoch": epoch, "phase": "eval_query", "task": i},
                log_all=True                                   # ← log every query episode
                # or: log_all=False, log_one_idx=some_index    # ← log just one query episode
            )
            
            after_win_rates.append(win_rate)
            after_rewards.append(np.mean(total_rewards))  # average over query episodes

        avg_pre_adapt = np.mean(pre_adapt_rewards)
        avg_eval_win_rate = np.mean(after_win_rates)
        avg_eval_rewards = np.mean(after_rewards)

        # print(f"\n[EVAL SUMMARY] Avg Reward: {avg_eval_rewards:.2f}, Avg Win Rate: {avg_eval_win_rate:.2f}")
        return avg_pre_adapt, avg_eval_win_rate, avg_eval_rewards

def save_checkpoint(policy: nn.Module, optimizer: optim.Optimizer, epoch: int, metric: float, path: str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "policy_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": metric,
        "torch_version": torch.__version__,
        "device": str(device),
    }, path)

def load_checkpoint(policy: nn.Module, optimizer: optim.Optimizer | None, path: str, map_location=None):
    if map_location is None:
        map_location = device
    ckpt = torch.load(path, map_location=map_location)
    policy.load_state_dict(ckpt["policy_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt

def switch_env(agent, host, port):
    agent.host = host
    agent.port = port
    agent.terminate_connection()
    agent.reconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="maml")
    parser.add_argument("--env", default=1, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--meta_epochs", default=1, type=int)   # 1000
    parser.add_argument("--logdir", default=path.join(path.dirname(path.abspath(__file__)), "logs"))
    parser.add_argument("--mlflow_url", default=None)
    parser.add_argument("--experiment_name", default="NetSecGame_MAML")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--checkpoint_dir", default=path.join(path.dirname(path.abspath(__file__)), "checkpoints"))
    parser.add_argument("--resume_from", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--eval_checkpoint", default=None, help="Path to checkpoint for evaluation-only mode")
    parser.add_argument("--best_by", default="winrate", choices=["winrate","metaloss"], help="Metric to track best model")

    args = parser.parse_args()

    makedirs(args.logdir, exist_ok=True)
    logging.basicConfig(filename=path.join(args.logdir, "maml_agent.log"),
                        filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO)

    if args.agent != "maml":
        raise ValueError("Only MAML agent is supported in this script.")

    request_trajectory = True

    # Hyperparameters for MAML
    inner_lr = 0.001
    outer_lr = 0.0002
    inner_steps = 3 # 3
    max_steps = 100
    
    # For training,
    # Per task: 30 episodes (support set, 10*3 inner steps) + 10 episodes (query set) = 40 episodes.
    # Per Epoch: 40 episodes * 5 tasks = 200 episodes. (run 5 epochs = 1,000 episodes)
    num_meta_batches = 5    # 5
    meta_batch_size = 10 	# 10
    test_batch_size = 10    # 10
    # For Evaluation,  
    # Per task: 75 episodes (support set, 25*3 inner steps) + 25 episodes (query set) = 100 episodes.
    # Per Epoch: 100 episodes * 5 tasks = 500 episodes. (run 1 evaluation = 500 episodes)
    # eval_meta_batch_size = 25	# for evaluation 
    # eval_test_batch_size = 25

    # Final agreed settings,
    # Per task: 30 episodes (support set, 10*3 inner steps) + 20 episodes (query set) = 50 episodes.
    # Per Epoch: 50 episodes * 5 tasks = 250 episodes. (run 1 evaluation = 250 episodes)
    
    # 9/17 additional run.
    # Per task: 150 episodes (support set, 50*3 inner steps) + 100 episodes (query set) = 250 episodes.
    # Per Epoch: 250 episodes * 1 tasks = 250 episodes. (run 1 evaluation = 250 episodes)
    
    num_eval_batches = 1
    eval_meta_batch_size = 50	# for evaluation 
    eval_test_batch_size = 100  

    # For performance comparison: DQN 
    # eval_meta_batch_size = 10	# for evaluation 
    # eval_test_batch_size = 10

    # Initialize MAML Agent
    policy = PolicyNetwork(12, (64, 64), 5).to(device)
    agent = MAMLAgent(
        host=args.host,
        port=args.port,
        role="Attacker",
        policy=policy,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        max_steps=max_steps,
        num_meta_batches=num_meta_batches,
        num_eval_batches=num_eval_batches,
        meta_batch_size=meta_batch_size,
        test_batch_size=test_batch_size,
        eval_meta_batch_size=eval_meta_batch_size,
        eval_test_batch_size=eval_test_batch_size,
        request_trajectory = request_trajectory
    )
    agent.register()
    print(f"[MAML Agent Connected] {args.host}:{args.port}")

    # Set up MLflow
    if args.mlflow_url:
        mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment(args.experiment_name)

    # Optionally resume training (weights + optimizer)
    if args.resume_from:
        ckpt = load_checkpoint(policy, agent.optimizer, args.resume_from)
        print(f"[Resume] Loaded checkpoint from {args.resume_from} (epoch={ckpt.get('epoch')}, best_metric={ckpt.get('best_metric')})")
    
    # for saving result as graph
    meta_losses = []
    before_adapt_returns = []
    after_adapt_returns = []
    eval_before_adapt_returns = []
    eval_after_adapt_returns = []
    eval_win_rates = []

    eval_every = 5   # run full evaluation only every 5 epochs
    eval_epochs = []   # to record which epochs we actually evaluated

    with mlflow.start_run(run_name=args.experiment_name) as run:
        # for log trajectory of evaluation episodes 
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")  # local time, no colons (Windows-safe)
        safe_exp = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" 
                        for ch in args.experiment_name)
        short_id = run.info.run_id[:8]  # keep a short suffix to avoid collisions
        run_dirname = f"{ts}_{safe_exp}_{short_id}"  # e.g., 20250815-143327_NetSecGame_MAML_ab12cd34
        traj_base_dir = path.join(args.logdir, "trajectories", run_dirname)
        traj_logger = TrajectoryLogger(traj_base_dir, run_id=run.info.run_id)
        print(f"[Trajectories] Writing to: {traj_base_dir}")
        agent._results_csv_path = path.join(traj_base_dir, "results", "eval_metrics.csv")
        makedirs(path.dirname(agent._results_csv_path), exist_ok=True)

        traj_logger.write_manifest({
            "schema_version": "v1",
            "run_id": run.info.run_id,
            "run_dirname": run_dirname,        # ← add this
            "agent": "MAML_Attacker_Agent",
            "feature_spec": agent.FEATURE_SPEC,
            "notes": "One evaluation trajectory per eval() call using log_one_idx."
        })

        if args.testing:
            print("=== Running Evaluation Only ===")
            if args.eval_checkpoint:
                ckpt = load_checkpoint(policy, None, args.eval_checkpoint)
                print(f"[Eval-Only] Loaded checkpoint from {args.eval_checkpoint} (epoch={ckpt.get('epoch')}, best_metric={ckpt.get('best_metric')})")

            eval_avg_before, win_rate, avg_reward = agent.evaluate(
                traj_logger=traj_logger, epoch=0, log_one_idx=test_batch_size - 1
            )
            mlflow.log_metric("eval_pre_adapt_avg_reward", eval_avg_before)
            mlflow.log_metric("eval_win_rate", win_rate)
            mlflow.log_metric("eval_avg_reward", avg_reward)
        else:
            print(f"=== Starting Meta-Training for {args.meta_epochs} Epochs ===")
            best_metric = -float('inf')  # higher is better; for meta-loss we’ll negate
            no_improve    = 0
            patience      = 5
            early_stopping = 0
            for epoch in range(1, args.meta_epochs + 1):
                t0 = time.time()
                # if torch.cuda.is_available(): # To check the resources
                #     torch.cuda.reset_peak_memory_stats()
                meta_loss, avg_before, avg_after = agent.outer_loop()

                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                #     print(f"[Epoch {epoch}] GPU alloc={torch.cuda.memory_allocated()/1024**2:.1f} MB | "
                #         f"reserved={torch.cuda.memory_reserved()/1024**2:.1f} MB | "
                #         f"peak_alloc={torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

                # print(f"[Epoch {epoch}] CPU={psutil.cpu_percent():.0f}% | "
                #     f"RAM={psutil.virtual_memory().used/1e9:.2f} GB | "
                #     f"elapsed={time.time()-t0:.2f}s")
                
                meta_losses.append(meta_loss)
                before_adapt_returns.append(avg_before[0])         # Pre-adaptation return
                after_adapt_returns.append(avg_after)          # Post-adaptation return
                
                mlflow.log_metric("meta_loss", meta_loss, step=epoch)
                
                for i, b in enumerate(avg_before):
                    mlflow.log_metric(f"return_before_step_{i}", b, step=epoch)

                # Only one post-adaptation return is available (after all inner steps)
                mlflow.log_metric("return_after_final", avg_after, step=epoch)

                print(f"[Epoch {epoch}] Meta Loss: {meta_loss:.4f}")
                print(f"[Summary Epoch {epoch}] Pre-Adapt Return: {avg_before[0]:.2f}, Post-Adapt Return: {avg_after:.2f}")

                # Save the model in every 10 epochs.
                if epoch % 10 == 0:
                    save_checkpoint(agent.policy, agent.optimizer, epoch, float('nan'),
                                    path.join(args.checkpoint_dir, f"last_epoch_{epoch:04d}.pth"))

                if epoch % eval_every == 0 or epoch == args.meta_epochs:
                    # --- Evaluation after each epoch ---
                    eval_avg_before, win_rate, eval_avg_reward = agent.evaluate(
                        traj_logger=traj_logger,
                        epoch=epoch,
                        log_one_idx=eval_test_batch_size - 1   # 0 = first query episode; or use test_batch_size-1 for "last"
                    )
                    eval_before_adapt_returns.append(eval_avg_before)
                    eval_after_adapt_returns.append(eval_avg_reward)
                    eval_win_rates.append(win_rate) 
                    eval_epochs.append(epoch)

                    mlflow.log_metric("eval_win_rate", win_rate, step=epoch)
                    mlflow.log_metric("eval_avg_reward", eval_avg_reward, step=epoch)
                    print(f"[Eval after Epoch {epoch}] initial avg reward: {eval_avg_before:.2f} Win Rate: {win_rate:.2f}%, Avg Reward: {eval_avg_reward:.2f} \n")
                    
                    if args.best_by == "winrate":
                        current_metric = win_rate  # higher is better
                    else:
                        current_metric = -meta_loss  # negate so higher is better

                    if current_metric > best_metric or epoch == args.meta_epochs:
                        best_metric = current_metric
                        ckpt_path = path.join(
                            args.checkpoint_dir,
                            f"best_epoch_{epoch:04d}_metric_{best_metric:.4f}.pth"
                        )
                        save_checkpoint(agent.policy, agent.optimizer, epoch, best_metric, ckpt_path)
                        print(f"* Saved BEST checkpoint at epoch {epoch} → {ckpt_path} (metric={best_metric:.4f})")
                        try:
                            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
                        except Exception:
                            pass
                        no_improve = 0
                    # else: # 10/08 it stopped to early. Temporary removed. 
                    #     no_improve += 1
                    #     if no_improve >= patience:
                    #         print(f"* Early stopping at epoch {epoch} (no improvement in {patience} evals).")
                    #         early_stopping = epoch
                    #         break


                if epoch % 50 == 0 or epoch == args.meta_epochs or epoch == early_stopping:
                    # --- 1) Meta-Loss Plot ---
                    plt.figure(figsize=(10, 5))
                    plt.plot(meta_losses, "--", color="black", label="Meta Loss")
                    plt.xlabel("Epoch"); plt.ylabel("Meta-Loss")
                    plt.title("Meta-Learning: Meta-Loss Over Epochs")
                    plt.grid(True); plt.legend(); plt.tight_layout()
                    meta_plot = path.join(args.logdir, f"meta_loss_plot_epoch{epoch}.png")
                    plt.savefig(meta_plot)
                    print(f"[Meta-Loss Plot Saved] → {meta_plot}")

                    # --- 2) Returns Plot ---
                    plt.figure(figsize=(10, 5))
                    plt.plot(before_adapt_returns,  color="blue",  label="Train Pre-Adapt")
                    plt.plot(after_adapt_returns,   color="green", label="Train Post-Adapt")
                    if eval_epochs:
                        plt.plot(eval_epochs, eval_before_adapt_returns, "o:", color="orange", label="Eval Pre-Adapt")
                        plt.plot(eval_epochs, eval_after_adapt_returns,  "o:", color="red",    label="Eval Post-Adapt")
                    plt.xlabel("Epoch"); plt.ylabel("Return")
                    plt.title("Meta-Learning: Returns Over Epochs")
                    plt.grid(True); plt.legend(); plt.tight_layout()
                    returns_plot = path.join(args.logdir, f"returns_plot_epoch{epoch}.png")
                    plt.savefig(returns_plot)
                    print(f"[Returns Plot Saved] → {returns_plot}")

                    # --- 3) Win-Rate Plot ---
                    plt.figure(figsize=(10, 5))
                    if eval_epochs:
                        plt.plot(eval_epochs, eval_win_rates, "x-.", color="purple", label="Eval Win Rate")
                    plt.xlabel("Epoch"); plt.ylabel("Win Rate (%)")
                    plt.title("Meta-Learning: Evaluation Win Rate")
                    plt.grid(True); plt.legend(); plt.tight_layout()
                    winrate_plot = path.join(args.logdir, f"winrate_plot_epoch{epoch}.png")
                    plt.savefig(winrate_plot)
                    print(f"[Win-Rate Plot Saved] → {winrate_plot}")

            print("=== Training Completed ===")

        print(f"Run saved at: {mlflow.get_tracking_uri()}, run_id: {run.info.run_id}")
        agent.terminate_connection()
