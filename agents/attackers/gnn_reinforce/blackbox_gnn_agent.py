"""
Factored (black-box) pure GNN agent for NetSecGame.

Decomposes action selection into up to four sequential sub-decisions:
  1. Action type        (global GNN embedding  → 5-way categorical)
  2. Source host        (host node embeddings  → categorical)
  3. Primary target     (typed node embeddings → categorical)
  4. Secondary param    (typed node embeddings → categorical, ExploitService and ExfiltrateData only)

Decision structure per action type:
  ScanNetwork:    source → target_network
  FindServices:   source → target_host
  FindData:       source → target_host
  ExploitService: source → target_host → target_service   (4 steps)
  ExfiltrateData: source → data        → target_host      (4 steps)

ExploitService needs the host-first ordering because the same service (e.g. ssh)
can run on multiple hosts; scoring the host first then the service on that host
avoids the ambiguity of a deduplicated service node representing many hosts.
ExfiltrateData needs an explicit destination head because there may be multiple
controlled hosts to exfiltrate to and the choice affects whether the win condition
is met (exfiltration must reach a public host).
"""

import argparse
from collections import deque
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None
from netsecgame import (
    Action,
    Observation,
    BaseAgent,
    generate_valid_actions,
    AgentRole,
    ActionType,
)
from netsecgame.game_components import AgentStatus
from policy_netsec import (
    ACTION_TYPE_LIST,
    ACTION_TARGET_PARAM,
    ACTION_TARGET_NODE_TYPE,
    ACTION_SECONDARY_PARAM,
    ACTION_SECONDARY_NODE_TYPE,
    state_to_pyg,
)
from utils import filter_log_files_from_state
from attempt_counts import AttemptCounts


# ── Policy ─────────────────────────────────────────────────────────────────


def _canonicalize_valid_actions(valid_actions: List[Action]) -> List[Action]:
    """Collapse exploit actions to one canonical service per (src, target host).

    In this environment the specific service choice does not change the outcome, so
    this agent treats ExploitService as a host-level decision and fills the service
    deterministically with the lexicographically smallest valid service for that host.
    """
    reduced: List[Action] = []
    exploit_idx_by_key: Dict[Tuple[object, object], int] = {}

    for action in valid_actions:
        if action.type != ActionType.ExploitService:
            reduced.append(action)
            continue

        key = (
            action.parameters.get("source_host"),
            action.parameters.get("target_host"),
        )
        existing_idx = exploit_idx_by_key.get(key)
        if existing_idx is None:
            exploit_idx_by_key[key] = len(reduced)
            reduced.append(action)
            continue

        existing = reduced[existing_idx]
        if str(action.parameters.get("target_service")) < str(
            existing.parameters.get("target_service")
        ):
            reduced[existing_idx] = action

    return reduced


class FactoredGNNPolicy(nn.Module):
    """
    Hierarchical GNN policy with four decoupled scoring heads:
      - type_head : global mean pool      → action type logits    (5-way)
      - src_head  : host_emb ‖ ctx        → source host score     (per host)
      - tgt_head  : node_emb ‖ ctx        → primary target score  (per node)
      - sec_head  : node_emb ‖ ctx ‖ tgt  → secondary score       (ExfiltrateData only)

    ExploitService:  source → target_host   (service filled deterministically)
    ExfiltrateData:  source → data        → target_host (destination)
    All others:      source → target                     (sec_head inactive, lp_sec = 0)
    """

    NODE_TYPES = ["network", "host", "service", "data"]
    EDGE_TYPES = [
        ("host", "in", "network"),
        ("network", "contains", "host"),
        ("host", "runs", "service"),
        ("host", "stores", "data"),
        ("service", "rev_runs", "host"),
        ("data", "rev_stores", "host"),
    ]

    def __init__(self, hidden_channels: int = 64, num_gnn_layers: int = 2):
        super().__init__()
        H = hidden_channels
        self.hidden_channels = H
        num_types = len(ACTION_TYPE_LIST)

        from torch_geometric.nn import Linear, GATv2Conv, HeteroConv

        # Node encoders (input feature dim = 16, matching state_to_pyg)
        self.node_encoders = nn.ModuleDict(
            {nt: Linear(16, H) for nt in self.NODE_TYPES}
        )

        # GNN backbone
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        et: GATv2Conv((-1, -1), H, add_self_loops=False)
                        for et in self.EDGE_TYPES
                    },
                    aggr="sum",
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Action type embedding used to condition src_head and tgt_head
        self.type_emb = nn.Embedding(num_types, H)

        # Head 1: action type from global graph summary + state summary
        # State summary has 3 features: controlled_no_data, data_not_exfiltrated,
        # exploitable_hosts — giving the type head phase-of-attack awareness.
        self.NUM_STATE_SUMMARY = 3
        self.type_head = nn.Sequential(
            nn.Linear(H + self.NUM_STATE_SUMMARY, H),
            nn.ReLU(),
            nn.Linear(H, num_types),
        )

        # Head 2: source host  — input: host_emb ‖ global_emb ‖ type_emb
        self.src_head = nn.Sequential(
            nn.Linear(H * 3, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

        # Head 3: target entity — input: node_emb ‖ global_emb ‖ type_emb ‖ src_emb
        self.tgt_head = nn.Sequential(
            nn.Linear(H * 4, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

        # Head 4: secondary parameter — input: node_emb ‖ global_emb ‖ type_emb ‖ src_emb ‖ tgt_emb
        # Active only for ExfiltrateData (destination after data) in this agent.
        self.sec_head = nn.Sequential(
            nn.Linear(H * 5, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

        # Value head — input: global_emb + state_summary → scalar state value
        self.value_head = nn.Sequential(
            nn.Linear(H + self.NUM_STATE_SUMMARY, H),
            nn.ReLU(),
            nn.Linear(H, 1),
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _gnn_forward(self, data) -> Dict[str, torch.Tensor]:
        h = {}
        for nt in self.NODE_TYPES:
            if nt in data.x_dict:
                h[nt] = F.relu(self.node_encoders[nt](data.x_dict[nt]))
        for conv in self.convs:
            active = {k: v for k, v in data.edge_index_dict.items() if v.numel() > 0}
            if active:
                h = conv(h, active)
            h = {nt: F.relu(emb) for nt, emb in h.items()}
        return h

    def _global_emb(self, h_dict: Dict[str, torch.Tensor], device) -> torch.Tensor:
        parts = [emb.mean(dim=0) for emb in h_dict.values() if emb.numel() > 0]
        return (
            torch.stack(parts).mean(dim=0)
            if parts
            else torch.zeros(self.hidden_channels, device=device)
        )

    def state_value(self, data, state_summary: torch.Tensor = None) -> torch.Tensor:
        """Compute V(s) from the global graph embedding (shares GNN backbone)."""
        h_dict = self._gnn_forward(data)
        device = next(self.parameters()).device
        g = self._global_emb(h_dict, device)
        if state_summary is None:
            state_summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)
        return self.value_head(torch.cat([g, state_summary])).squeeze(-1)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        data,
        object_to_idx: Dict,
        valid_actions: List[Action],
        temperature: float = 1.0,
        state_summary: torch.Tensor = None,
    ) -> Tuple[Action, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (action, total_log_prob, total_entropy).

        Decision order:
          1. action type       (always)
          2. source host       (always)
          3. primary target    (always: target_host for Exploit/FindServices/FindData/Scan,
                                         data for Exfiltrate)
          4. secondary param   (ExfiltrateData only: destination host for chosen data)

        Temperature is always applied to logits. During training, pass a value
        that anneals from 1.0 toward a non-trivial floor (e.g. 0.5) to preserve
        exploration. During evaluation, use a lower value (e.g. 0.1) for
        near-greedy behavior that avoids pathological argmax lock-in of factored
        action spaces.
        """
        device = next(self.parameters()).device
        zero = torch.zeros(self.hidden_channels, device=device)
        temperature = max(temperature, 1e-8)

        def _fallback_uniform_action() -> Tuple[Action, None, None]:
            # If factorization cannot represent current valid actions, execute a
            # behavior action without injecting biased policy-gradient terms.
            return random.choice(valid_actions), None, None

        h_dict = self._gnn_forward(data)
        g = self._global_emb(h_dict, device)  # [H]
        if state_summary is None:
            state_summary = torch.zeros(self.NUM_STATE_SUMMARY, device=device)

        # ── 1. Action type ────────────────────────────────────────────────
        valid_type_set = {a.type for a in valid_actions}
        type_mask = torch.tensor(
            [at in valid_type_set for at in ACTION_TYPE_LIST],
            dtype=torch.bool,
            device=device,
        )
        type_input = torch.cat([g, state_summary])
        type_logits = self.type_head(type_input).masked_fill(~type_mask, -1e9)
        type_logits = type_logits / temperature
        type_dist = Categorical(logits=type_logits)
        type_idx = type_dist.sample().item()
        chosen_type = ACTION_TYPE_LIST[type_idx]
        t_idx_t = torch.tensor(type_idx, device=device)
        t_emb = self.type_emb(t_idx_t)  # [H]
        lp_type = type_dist.log_prob(t_idx_t)
        ent_type = type_dist.entropy()

        candidates = [a for a in valid_actions if a.type == chosen_type]

        # ── 2. Source host ────────────────────────────────────────────────
        lp_src = torch.zeros((), device=device)
        ent_src = torch.zeros((), device=device)
        src_emb = zero

        valid_srcs = {
            a.parameters["source_host"]
            for a in candidates
            if "source_host" in a.parameters
        }
        if valid_srcs:
            host_embs = h_dict.get(
                "host", torch.empty(0, self.hidden_channels, device=device)
            )
            scores, src_keys = [], []
            for obj, idx in object_to_idx.get("host", {}).items():
                if obj in valid_srcs:
                    inp = torch.cat([host_embs[idx], g, t_emb])
                    scores.append(self.src_head(inp).squeeze(-1))
                    src_keys.append((obj, idx))

            if scores:
                src_logits = torch.stack(scores)
                src_logits = src_logits / temperature
                src_dist = Categorical(logits=src_logits)
                s_idx = src_dist.sample().item()
                s_idx_t = torch.tensor(s_idx, device=device)
                chosen_src, src_nidx = src_keys[s_idx]
                src_emb = host_embs[src_nidx]
                lp_src = src_dist.log_prob(s_idx_t)
                ent_src = src_dist.entropy()
                candidates = [
                    a
                    for a in candidates
                    if a.parameters.get("source_host") == chosen_src
                ]

        # ── 3. Primary target ─────────────────────────────────────────────
        tgt_param = ACTION_TARGET_PARAM.get(chosen_type)
        tgt_node_type = ACTION_TARGET_NODE_TYPE.get(chosen_type)

        if tgt_param is None or tgt_node_type is None:
            return _fallback_uniform_action()

        valid_tgts = {
            a.parameters[tgt_param] for a in candidates if tgt_param in a.parameters
        }
        tgt_embs = h_dict.get(
            tgt_node_type, torch.empty(0, self.hidden_channels, device=device)
        )
        scores, tgt_keys = [], []
        for obj, idx in object_to_idx.get(tgt_node_type, {}).items():
            if obj in valid_tgts:
                inp = torch.cat([tgt_embs[idx], g, t_emb, src_emb])
                scores.append(self.tgt_head(inp).squeeze(-1))
                tgt_keys.append((obj, idx))

        if not scores:
            return _fallback_uniform_action()

        tgt_logits = torch.stack(scores)
        tgt_logits = tgt_logits / temperature
        tgt_dist = Categorical(logits=tgt_logits)
        t2_idx = tgt_dist.sample().item()
        t2_idx_t = torch.tensor(t2_idx, device=device)
        chosen_tgt, chosen_tgt_nidx = tgt_keys[t2_idx]
        tgt_emb = tgt_embs[chosen_tgt_nidx]  # [H] — passed to sec_head when active
        lp_tgt = tgt_dist.log_prob(t2_idx_t)
        ent_tgt = tgt_dist.entropy()

        candidates = [
            a for a in candidates if a.parameters.get(tgt_param) == chosen_tgt
        ]

        # ── 4. Secondary parameter (ExfiltrateData only) ─
        lp_sec = torch.zeros((), device=device)
        ent_sec = torch.zeros((), device=device)

        sec_param = None
        sec_node_type = None
        if chosen_type == ActionType.ExfiltrateData:
            sec_param = ACTION_SECONDARY_PARAM.get(chosen_type)
            sec_node_type = ACTION_SECONDARY_NODE_TYPE.get(chosen_type)

        if sec_param is not None and sec_node_type is not None:
            valid_secs = {
                a.parameters[sec_param] for a in candidates if sec_param in a.parameters
            }
            sec_embs = h_dict.get(
                sec_node_type, torch.empty(0, self.hidden_channels, device=device)
            )
            scores, sec_keys = [], []
            for obj, idx in object_to_idx.get(sec_node_type, {}).items():
                if obj in valid_secs:
                    inp = torch.cat([sec_embs[idx], g, t_emb, src_emb, tgt_emb])
                    scores.append(self.sec_head(inp).squeeze(-1))
                    sec_keys.append(obj)

            if scores:
                sec_logits = torch.stack(scores)
                sec_logits = sec_logits / temperature
                sec_dist = Categorical(logits=sec_logits)
                s2_idx = sec_dist.sample().item()
                s2_idx_t = torch.tensor(s2_idx, device=device)
                chosen_sec = sec_keys[s2_idx]
                lp_sec = sec_dist.log_prob(s2_idx_t)
                ent_sec = sec_dist.entropy()
                candidates = [
                    a for a in candidates if a.parameters.get(sec_param) == chosen_sec
                ]

        if len(candidates) != 1:
            return _fallback_uniform_action()
        action = candidates[0]
        return (
            action,
            lp_type + lp_src + lp_tgt + lp_sec,
            ent_type + ent_src + ent_tgt + ent_sec,
        )


# ── Agent ──────────────────────────────────────────────────────────────────


class BlackBoxGNNAgent(BaseAgent):
    def __init__(self, host, port, role, seed, weights_file="best_blackbox_gnn.pth"):
        super().__init__(host, port, role)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.policy = FactoredGNNPolicy()
        self.weights_file = weights_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.attempt_counts = AttemptCounts()

    def reset_episode_tracking(self):
        self.attempt_counts.reset()

    def record_action_attempt(self, action: Action):
        self.attempt_counts.record(action)

    def load_weights(self):
        if not os.path.exists(self.weights_file):
            raise FileNotFoundError(f"Weights file {self.weights_file} not found.")
        self.policy.load_state_dict(
            torch.load(self.weights_file, weights_only=False, map_location=self.device)
        )
        print(f"Loaded weights from {self.weights_file}")

    def _state_summary(self, state) -> torch.Tensor:
        """Compute phase-of-attack summary features from game state.

        Returns a 3-dim tensor:
          [0] controlled hosts with no known data  (need FindData)
          [1] known data not yet exfiltrated        (need ExfiltrateData)
          [2] uncontrolled hosts with known services (can ExploitService)

        All values are normalized to roughly [0, 1].
        """
        controlled = state.controlled_hosts
        # Controlled hosts where no data has been found yet
        controlled_no_data = sum(
            1 for h in controlled if not state.known_data.get(h)
        )
        # Data items on hosts that are not controlled+public (not yet exfiltrated)
        data_not_exfiltrated = 0
        for host, data_set in state.known_data.items():
            if host not in controlled:
                data_not_exfiltrated += len(data_set)
        # Uncontrolled hosts with known services (exploitable)
        exploitable = sum(
            1 for h in state.known_services
            if h not in controlled and state.known_services[h]
        )
        return torch.tensor(
            [
                min(controlled_no_data, 10) / 10.0,
                min(data_not_exfiltrated, 10) / 10.0,
                min(exploitable, 10) / 10.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def select_action(
        self,
        observation: Observation,
        temperature: float = 1.0,
    ) -> Tuple[Action, Optional[torch.Tensor], Optional[torch.Tensor]]:
        valid_actions = generate_valid_actions(observation.state, include_blocks=False)
        valid_actions = _canonicalize_valid_actions(valid_actions)
        if not valid_actions:
            raise ValueError("No valid actions available.")
        pyg_data, object_to_idx, _ = state_to_pyg(
            observation.state,
            attempt_counts=self.attempt_counts,
        )
        summary = self._state_summary(observation.state)
        return self.policy(
            pyg_data.to(self.device),
            object_to_idx,
            valid_actions,
            temperature=temperature,
            state_summary=summary,
        )

    def compute_value(self, observation: Observation) -> torch.Tensor:
        pyg_data, _, _ = state_to_pyg(
            observation.state,
            attempt_counts=self.attempt_counts,
        )
        summary = self._state_summary(observation.state)
        return self.policy.state_value(pyg_data.to(self.device), state_summary=summary)


# ── Helpers ────────────────────────────────────────────────────────────────


def compute_returns(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns (unnormalized — needed as value targets)."""
    returns, R = [], 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


def _rolling_win_rate(win_history: deque) -> float:
    """Mean of the last N win outcomes (1=win, 0=other) stored in win_history."""
    if not win_history:
        return 0.0
    return sum(win_history) / len(win_history)


def evaluate_policy(
    agent: BlackBoxGNNAgent,
    eval_episodes: int,
    verbose: bool = False,
    temperature: float = 0.1,
) -> Tuple[float, float, float]:
    """Evaluate policy using low-temperature sampling.

    Pure argmax (greedy) fails with factored action spaces because the
    independently optimal choice at each decision step (type → source →
    target → secondary) does not compose into a globally optimal action.
    Low-temperature sampling keeps the policy nearly deterministic while
    avoiding pathological lock-in on a single action type.
    """
    agent.policy.eval()
    wins = 0
    total_steps = 0
    total_rewards = 0.0
    with torch.no_grad():
        for e in range(1, eval_episodes + 1):
            obs = agent.request_game_reset()
            agent.reset_episode_tracking()
            obs = filter_log_files_from_state(obs)
            num_steps = 0
            total_reward = 0.0
            while obs and not obs.end:
                action, _, _ = agent.select_action(
                    obs,
                    temperature=temperature,
                )
                if verbose:
                    print(f"    Step {num_steps+1}: {action}")
                agent.record_action_attempt(action)
                obs = agent.make_step(action)
                obs = filter_log_files_from_state(obs)
                total_reward += obs.reward
                num_steps += 1
                if verbose:
                    print(f"      → reward={obs.reward}, end={obs.end}")
            end_status = obs.info.get("end_reason") if obs.info else None
            if end_status == AgentStatus.Success:
                wins += 1
            total_steps += num_steps
            total_rewards += total_reward
            if verbose:
                print(
                    f"  [Eval] Ep {e} | Steps: {num_steps} | R: {total_reward:.2f} | {end_status}"
                )
    agent.policy.train()
    return (
        wins / eval_episodes,
        total_steps / eval_episodes,
        total_rewards / eval_episodes,
    )


# ── Training loop ──────────────────────────────────────────────────────────


def run_agent(args):
    agent = BlackBoxGNNAgent(
        args.host,
        args.port,
        AgentRole.Attacker,
        seed=args.seed,
        weights_file=args.weights,
    )

    try:
        observation = agent.register()
        agent.reset_episode_tracking()
        observation = filter_log_files_from_state(observation)
    except Exception as e:
        print(f"Failed to register: {e}")
        return

    if args.wandb:
        run_name = (
            f"gnn-lr={args.lr}_ent={args.entropy_beta}"
            f"_cur={args.curiosity_weight}_seed={args.seed}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )
    if args.eval:
        print(
            f"\n--- Evaluation Mode: {args.episodes} episodes "
            f"(temp={args.eval_temperature}) ---"
        )
        agent.load_weights()
        agent.policy.eval()
        win_rate, avg_steps, avg_reward = evaluate_policy(
            agent,
            args.episodes,
            args.verbose,
            temperature=args.eval_temperature,
        )
        print(
            f"--- Win Rate: {win_rate * 100:.2f}% | "
            f"Avg Steps: {avg_steps:.2f} | "
            f"Avg Reward: {avg_reward:.2f} ---"
        )
        if args.wandb:
            wandb.finish()
        agent.terminate_connection()
        return

    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-6,
    )
    best_win_rate = -1.0
    wins, detected, timeouts = 0, 0, 0
    win_history: deque = deque(maxlen=100)

    wandb_watched = False
    batch_loss = torch.zeros((), device=agent.device)
    batch_policy_loss = 0.0
    batch_value_loss = 0.0
    batch_entropy_bonus = 0.0
    batch_has_grad = False
    optimizer.zero_grad()

    for episode in range(1, args.episodes + 1):
        log_probs, rewards, entropies, values = [], [], [], []
        num_steps = 0
        fallback_steps = 0

        # Entropy annealing: geometric decay from entropy_beta → entropy_min
        current_entropy_beta = (
            args.entropy_beta
            * (args.entropy_min / args.entropy_beta) ** (episode / args.episodes)
            if args.entropy_beta > args.entropy_min
            else args.entropy_beta
        )
        # Training temperature: anneal from train_temperature_start →
        # train_temperature_end over train_temperature_anneal_frac of training,
        # then hold at train_temperature_end for the remainder.
        train_temp_end = (
            args.train_temperature_end
            if args.train_temperature_end is not None
            else args.eval_temperature
        )
        anneal_episodes = max(1, int(args.train_temperature_anneal_frac * args.episodes))
        anneal_progress = min(episode / anneal_episodes, 1.0)
        train_temperature = args.train_temperature_start + (
            train_temp_end - args.train_temperature_start
        ) * anneal_progress
        # Curiosity weight: linear anneal to zero over curiosity_anneal_frac of training
        curiosity_anneal_episode = int(args.curiosity_anneal_frac * args.episodes)
        if episode < curiosity_anneal_episode:
            current_curiosity = args.curiosity_weight * (
                1.0 - episode / curiosity_anneal_episode
            )
        else:
            current_curiosity = 0.0

        prev_hosts = len(observation.state.known_hosts)
        prev_services = sum(len(s) for s in observation.state.known_services.values())
        prev_data = sum(len(d) for d in observation.state.known_data.values())

        while observation and not observation.end:
            values.append(agent.compute_value(observation))
            action, log_prob, entropy = agent.select_action(
                observation, temperature=train_temperature
            )

            if args.wandb and not wandb_watched:
                try:
                    wandb.watch(agent.policy, log="all", log_freq=args.eval_interval)
                except ValueError:
                    # GATv2Conv(-1,-1) uses lazy init; some params may still
                    # be uninitialized if their edge types were absent in this
                    # graph. Skip wandb.watch — metrics are logged separately.
                    print(
                        "Warning: wandb.watch skipped (lazy params not yet initialized)"
                    )
                wandb_watched = True

            if args.verbose:
                print(
                    f"[Ep {episode} | Step {num_steps}] {action.type} {action.parameters}"
                )

            agent.record_action_attempt(action)
            observation = agent.make_step(action)
            observation = filter_log_files_from_state(observation)
            num_steps += 1

            # Curiosity bonus for newly discovered entities
            curiosity_bonus = 0.0
            cur_hosts = len(observation.state.known_hosts)
            cur_services = sum(
                len(s) for s in observation.state.known_services.values()
            )
            cur_data = sum(len(d) for d in observation.state.known_data.values())
            if cur_hosts > prev_hosts:
                curiosity_bonus += current_curiosity
                prev_hosts = cur_hosts
            if cur_services > prev_services:
                curiosity_bonus += current_curiosity
                prev_services = cur_services
            if cur_data > prev_data:
                curiosity_bonus += current_curiosity
                prev_data = cur_data

            rewards.append(observation.reward + curiosity_bonus)
            if log_prob is None:
                fallback_steps += 1
            log_probs.append(log_prob)
            entropies.append(entropy)

        end_status = observation.info.get("end_reason") if observation.info else None
        if end_status == AgentStatus.Success:
            wins += 1
        elif end_status == AgentStatus.Fail:
            detected += 1
        else:
            timeouts += 1
        win_history.append(1 if end_status == AgentStatus.Success else 0)

        # REINFORCE update with value baseline (accumulated over batch)
        if log_probs:
            returns = compute_returns(rewards).to(agent.device)
            v_preds = torch.stack(values)
            advantages = returns - v_preds.detach()

            policy_terms = [
                -lp * advantages[i]
                for i, lp in enumerate(log_probs)
                if lp is not None
            ]
            policy_loss = (
                torch.stack(policy_terms).sum()
                if policy_terms
                else torch.zeros((), device=agent.device)
            )
            value_loss = F.mse_loss(v_preds, returns)
            entropy_terms = [ent for ent in entropies if ent is not None]
            entropy_bonus = (
                current_entropy_beta * torch.stack(entropy_terms).mean()
                if entropy_terms
                else torch.zeros((), device=agent.device)
            )
            loss = (policy_loss + 0.5 * value_loss - entropy_bonus) / args.batch_size

            if loss.requires_grad:
                loss.backward()
                batch_has_grad = True

            batch_loss = batch_loss + loss.detach()
            batch_policy_loss += policy_loss.item() / args.batch_size
            batch_value_loss += value_loss.item() / args.batch_size
            batch_entropy_bonus += entropy_bonus.item() / args.batch_size

            print(
                f"Ep {episode:05d} | Steps: {num_steps:03d} | R: {sum(rewards):7.2f} | "
                f"Loss: {loss.item() * args.batch_size:8.3f} | VLoss: {value_loss.item():6.3f} | "
                f"EntBeta: {current_entropy_beta:.5f} | "
                f"Temp: {train_temperature:.3f} | "
                f"Cur: {current_curiosity:.2f} | "
                f"Fallback: {fallback_steps}/{num_steps} | {end_status}"
            )
            if args.wandb:
                wandb.log(
                    {
                        "train/loss": loss.item() * args.batch_size,
                        "train/policy_loss": policy_loss.item(),
                        "train/value_loss": value_loss.item(),
                        "train/entropy_bonus": entropy_bonus.item(),
                        "train/episode_reward": sum(rewards),
                        "train/steps": num_steps,
                        "train/entropy_beta": current_entropy_beta,
                        "train/curiosity_weight": current_curiosity,
                        "train/temperature": train_temperature,
                        "train/fallback_steps": fallback_steps,
                        "train/fallback_ratio": (
                            fallback_steps / num_steps if num_steps > 0 else 0.0
                        ),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/win_rate": _rolling_win_rate(win_history),
                    },
                    step=episode,
                )

        # Step optimizer every batch_size episodes
        if episode % args.batch_size == 0:
            if batch_has_grad:
                torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            batch_loss = torch.zeros((), device=agent.device)
            batch_policy_loss = 0.0
            batch_value_loss = 0.0
            batch_entropy_bonus = 0.0
            batch_has_grad = False

        # Periodic checkpoint
        if args.checkpoint_interval and episode % args.checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = os.path.join("checkpoints", f"checkpoint_ep{episode}.pth")
            torch.save(agent.policy.state_dict(), ckpt_path)
            print(f"--> Checkpoint saved: {ckpt_path}")

        if episode % args.eval_interval == 0:
            print(f"\n--- Eval @ Episode {episode} ---")
            win_rate, avg_steps, avg_reward = evaluate_policy(
                agent,
                args.eval_episodes,
                args.verbose,
                temperature=args.eval_temperature,
            )
            print(
                f"--- Win Rate: {win_rate * 100:.2f}% | "
                f"Avg Steps: {avg_steps:.1f} | "
                f"Avg Reward: {avg_reward:.2f} ---\n"
            )
            scheduler.step(win_rate)
            if args.wandb:
                wandb.log(
                    {
                        "eval/win_rate": win_rate,
                        "eval/avg_steps": avg_steps,
                        "eval/avg_reward": avg_reward,
                    },
                    step=episode,
                )
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(agent.policy.state_dict(), args.weights)
                print(
                    f"--> New best ({best_win_rate * 100:.1f}%). Saved to {args.weights}"
                )
                if best_win_rate >= 1.0:
                    print("[Early Stop] 100% eval win rate.")
                    break

        if episode < args.episodes:
            observation = agent.request_game_reset()
            agent.reset_episode_tracking()
            observation = filter_log_files_from_state(observation)

    # Flush any remaining partial batch
    if batch_has_grad:
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if args.wandb:
        wandb.finish()
    agent.terminate_connection()
    print(f"\n--- Summary ({args.episodes} episodes) ---")
    print(f"Wins: {wins} | Detected: {detected} | Timeouts: {timeouts}")
    print(f"Win Rate: {wins / args.episodes * 100:.2f}%")


# ── Entry point ────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Factored black-box GNN agent for NetSecGame"
    )
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=9000, type=int)
    parser.add_argument("--episodes", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weights", default="best_blackbox_gnn.pth", type=str)
    parser.add_argument(
        "--entropy_beta", default=0.1, type=float, help="Initial entropy coefficient"
    )
    parser.add_argument(
        "--entropy_min",
        default=0.01,
        type=float,
        help="Final entropy coefficient after annealing",
    )
    parser.add_argument(
        "--curiosity_weight",
        default=2.0,
        type=float,
        help="Reward bonus per novel entity discovered",
    )
    parser.add_argument(
        "--curiosity_anneal_frac",
        default=0.5,
        type=float,
        help="Fraction of total episodes over which curiosity anneals to zero",
    )
    parser.add_argument(
        "--eval_interval", default=200, type=int, help="Episodes between eval runs"
    )
    parser.add_argument(
        "--eval_episodes", default=100, type=int, help="Episodes per eval run"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run in evaluation mode (requires --weights)",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Number of episodes to accumulate gradients over before optimizer step",
    )
    parser.add_argument(
        "--eval_temperature",
        default=0.1,
        type=float,
        help="Softmax temperature for eval (low = near-greedy, avoids argmax lock-in)",
    )
    parser.add_argument(
        "--train_temperature_start",
        default=1.0,
        type=float,
        help="Initial softmax temperature for training annealed toward eval_temperature",
    )
    parser.add_argument(
        "--train_temperature_end",
        default=None,
        type=float,
        help="Final training softmax temperature (defaults to eval_temperature if unset)",
    )
    parser.add_argument(
        "--train_temperature_anneal_frac",
        default=1.0,
        type=float,
        help="Fraction of training over which train temperature anneals to train_temperature_end",
    )
    parser.add_argument(
        "--checkpoint_interval",
        default=5000,
        type=int,
        help="Save a checkpoint every N episodes (0 to disable)",
    )
    parser.add_argument("--verbose", action="store_true")
    # ── wandb ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        default="sgrl-sec-blackbox-gnn",
        type=str,
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        type=str,
        help="wandb entity/team (optional)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_agent(args)
