"""
Tests for FactoredGNNPolicy and state_to_pyg.

Covers:
  - state_to_pyg correctness (determinism, deduplication, completeness)
  - Policy output is always a valid action
  - log_prob.requires_grad (training mode, REINFORCE won't crash)
  - ExploitService: returned service is on the chosen target host
  - ExploitService actions are canonicalized to a host-level choice
  - ExfiltrateData: returned data is on the source host; destination is scored
  - Simple action types (Scan, FindServices, FindData) work without sec_head

Run with:
    python -m pytest test_blackbox_gnn.py -v
"""

import argparse
from collections import deque
import sys
import pytest
import torch
from netsecgame.game_components import (
    GameState, IP, Network, Service, Data, ActionType, Action,
)
from netsecgame import generate_valid_actions
from policy_netsec import state_to_pyg
from blackbox_pure_gnn_agent import FactoredGNNPolicy, _canonicalize_valid_actions
from policy_netsec import ACTION_SECONDARY_PARAM, ACTION_SECONDARY_NODE_TYPE


# ── Common game objects ────────────────────────────────────────────────────

SSH  = Service(name="ssh",  type="tcp", version="8.0",  is_local=False)
HTTP = Service(name="http", type="tcp", version="2.4",  is_local=False)
RDP  = Service(name="rdp",  type="tcp", version="10.0", is_local=False)

CREDS   = Data(owner="Admin", id="creds")
WEBDATA = Data(owner="Admin", id="webdata")


# ── Helpers ────────────────────────────────────────────────────────────────

def make_state(
    controlled=None,
    hosts=None,
    networks=None,
    services=None,
    data=None,
):
    controlled = controlled or {IP("10.0.0.1")}
    hosts      = hosts      or set()
    networks   = networks   or {Network("10.0.0.0", 24)}
    services   = services   or {}
    data       = data       or {}
    return GameState(
        controlled_hosts=controlled,
        known_hosts=hosts | controlled,
        known_services=services,
        known_data=data,
        known_networks=networks,
        known_blocks={},
    )


def run_policy(policy, state, n=30, temperature=1.0):
    """Run the policy n times against state, return list of (action, lp, ent)."""
    valid   = generate_valid_actions(state, include_blocks=False)
    pyg, obj_idx, _ = state_to_pyg(state)
    return [policy(pyg, obj_idx, valid, temperature=temperature) for _ in range(n)], valid



# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def policy():
    torch.manual_seed(0)
    # Small hidden size so tests run fast
    return FactoredGNNPolicy(hidden_channels=16, num_gnn_layers=1)


# ══════════════════════════════════════════════════════════════════════════
# state_to_pyg
# ══════════════════════════════════════════════════════════════════════════

class TestStateToPyg:

    def test_deterministic_ordering(self):
        """Calling state_to_pyg twice on the same state gives identical mappings."""
        state = make_state(
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {SSH, HTTP},
            },
        )
        _, idx1, inv1 = state_to_pyg(state)
        _, idx2, inv2 = state_to_pyg(state)
        assert idx1 == idx2
        assert inv1 == inv2

    def test_all_hosts_in_graph(self):
        """Every known and controlled host must appear as a node."""
        state = make_state(
            controlled={IP("10.0.0.1"), IP("10.0.0.2")},
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
        )
        _, obj_idx, _ = state_to_pyg(state)
        all_hosts = state.known_hosts | state.controlled_hosts
        for h in all_hosts:
            assert h in obj_idx["host"], f"{h} missing from graph"

    def test_service_deduplication_same_service(self):
        """The same service on two hosts produces exactly one service node."""
        state = make_state(
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {SSH},   # identical Service object
            },
        )
        _, obj_idx, _ = state_to_pyg(state)
        assert len(obj_idx["service"]) == 1

    def test_distinct_services_not_deduplicated(self):
        """Different services on different hosts both appear as separate nodes."""
        state = make_state(
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {HTTP},
            },
        )
        _, obj_idx, _ = state_to_pyg(state)
        assert len(obj_idx["service"]) == 2

    def test_multiple_services_same_host(self):
        """Multiple services on one host all appear as nodes."""
        state = make_state(
            hosts={IP("192.168.1.1")},
            services={IP("192.168.1.1"): {SSH, HTTP, RDP}},
        )
        _, obj_idx, _ = state_to_pyg(state)
        assert len(obj_idx["service"]) == 3

    def test_multiple_data_same_host(self):
        """Multiple data objects on the same host all appear as separate nodes."""
        state = make_state(data={IP("10.0.0.1"): {CREDS, WEBDATA}})
        _, obj_idx, _ = state_to_pyg(state)
        assert len(obj_idx["data"]) == 2

    def test_node_feature_dim(self):
        """All node feature matrices must have exactly 16 columns."""
        state = make_state(
            hosts={IP("192.168.1.1")},
            services={IP("192.168.1.1"): {SSH}},
            data={IP("10.0.0.1"): {CREDS}},
        )
        pyg, _, _ = state_to_pyg(state)
        for nt, feat in pyg.x_dict.items():
            assert feat.shape[1] == 16, f"{nt} features have wrong dim {feat.shape}"

    def test_controlled_host_feature(self):
        """Controlled hosts must have feature[0] == 1; uncontrolled == 0."""
        controlled = IP("10.0.0.1")
        uncontrolled = IP("192.168.1.1")
        state = make_state(
            controlled={controlled},
            hosts={uncontrolled},
        )
        pyg, obj_idx, _ = state_to_pyg(state)
        feats = pyg["host"].x
        assert feats[obj_idx["host"][controlled]][0].item() == 1.0
        assert feats[obj_idx["host"][uncontrolled]][0].item() == 0.0

    def test_host_findservices_attempts_feature(self):
        """Host feature[2] tracks findservices attempts via AttemptCounts."""
        from attempt_counts import AttemptCounts
        host = IP("192.168.1.1")
        state = make_state(hosts={host})
        counts = AttemptCounts()
        counts.findservices[host] = 2
        pyg, obj_idx, _ = state_to_pyg(state, attempt_counts=counts)
        feats = pyg["host"].x
        assert feats[obj_idx["host"][host]][2].item() == pytest.approx(0.2)

    def test_host_finddata_attempts_feature(self):
        """Host feature[3] tracks finddata attempts."""
        from attempt_counts import AttemptCounts
        host = IP("192.168.1.1")
        state = make_state(hosts={host})
        counts = AttemptCounts()
        counts.finddata[host] = 5
        pyg, obj_idx, _ = state_to_pyg(state, attempt_counts=counts)
        feats = pyg["host"].x
        assert feats[obj_idx["host"][host]][3].item() == pytest.approx(0.5)

    def test_host_exploit_attempts_feature(self):
        """Host feature[4] tracks exploit attempts."""
        from attempt_counts import AttemptCounts
        host = IP("192.168.1.1")
        state = make_state(hosts={host})
        counts = AttemptCounts()
        counts.exploit[host] = 7
        pyg, obj_idx, _ = state_to_pyg(state, attempt_counts=counts)
        feats = pyg["host"].x
        assert feats[obj_idx["host"][host]][4].item() == pytest.approx(0.7)

    def test_host_attempts_saturate_at_ten(self):
        """Counts above 10 clamp to 1.0 (normalized)."""
        from attempt_counts import AttemptCounts
        host = IP("192.168.1.1")
        state = make_state(hosts={host})
        counts = AttemptCounts()
        counts.exploit[host] = 50
        pyg, obj_idx, _ = state_to_pyg(state, attempt_counts=counts)
        feats = pyg["host"].x
        assert feats[obj_idx["host"][host]][4].item() == pytest.approx(1.0)

    def test_host_attempts_none_defaults_to_zero(self):
        """When attempt_counts is None, all attempt-related features are 0."""
        host = IP("192.168.1.1")
        state = make_state(hosts={host})
        pyg, obj_idx, _ = state_to_pyg(state)
        feats = pyg["host"].x
        assert feats[obj_idx["host"][host]][2].item() == 0.0
        assert feats[obj_idx["host"][host]][3].item() == 0.0
        assert feats[obj_idx["host"][host]][4].item() == 0.0

    def test_network_scan_attempts_feature(self):
        """Network feature[0] tracks scan attempts."""
        from attempt_counts import AttemptCounts
        net = Network("10.0.0.0", 24)
        state = make_state(networks={net})
        counts = AttemptCounts()
        counts.scan[net] = 4
        pyg, obj_idx, _ = state_to_pyg(state, attempt_counts=counts)
        feats = pyg["network"].x
        assert feats[obj_idx["network"][net]][0].item() == pytest.approx(0.4)

    def test_network_feature_zero_when_no_counts(self):
        """Network scan-attempt feature is zero when attempt_counts is None."""
        net = Network("10.0.0.0", 24)
        state = make_state(networks={net})
        pyg, obj_idx, _ = state_to_pyg(state)
        feats = pyg["network"].x
        assert feats[obj_idx["network"][net]][0].item() == 0.0

    def test_network_has_yielded_hosts_feature(self):
        """feat[1] flips to 1.0 once a known host lies inside the network's CIDR."""
        net_empty = Network("192.168.99.0", 24)
        net_populated = Network("10.0.0.0", 24)
        state = make_state(networks={net_empty, net_populated})
        pyg, obj_idx, _ = state_to_pyg(state)
        feats = pyg["network"].x
        assert feats[obj_idx["network"][net_empty]][1].item() == 0.0
        assert feats[obj_idx["network"][net_populated]][1].item() == 1.0

    def test_network_known_host_count_feature(self):
        """feat[2] encodes normalized count of known hosts inside the CIDR."""
        net = Network("10.0.0.0", 24)
        state = make_state(
            controlled={IP("10.0.0.1")},
            hosts={IP("10.0.0.2"), IP("10.0.0.3")},
            networks={net},
        )
        pyg, obj_idx, _ = state_to_pyg(state)
        feats = pyg["network"].x
        # 3 known hosts in CIDR (controlled ∪ hosts), _norm(3) = 0.3
        assert feats[obj_idx["network"][net]][2].item() == pytest.approx(0.3)

    def test_data_exfil_attempts_feature(self):
        """Data feature[2] tracks exfil attempts per data node."""
        from attempt_counts import AttemptCounts
        host = IP("10.0.0.1")
        d = Data(owner="Admin", id="creds")
        state = make_state(
            controlled={host},
            data={host: {d}},
        )
        counts = AttemptCounts()
        counts.exfil[d] = 6
        pyg, obj_idx, _ = state_to_pyg(state, attempt_counts=counts)
        feats = pyg["data"].x
        assert feats[obj_idx["data"][d]][2].item() == pytest.approx(0.6)


# ══════════════════════════════════════════════════════════════════════════
# General policy output
# ══════════════════════════════════════════════════════════════════════════

class TestPolicyOutputBasics:

    def _rich_state(self):
        """State with all five action types available."""
        return make_state(
            controlled={IP("10.0.0.1")},
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            networks={Network("192.168.1.0", 24), Network("10.0.0.0", 24)},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {SSH, HTTP},
            },
            data={IP("192.168.1.1"): {CREDS}},
        )

    def test_output_always_in_valid_actions(self, policy):
        """Every action returned by the policy must be in valid_actions."""
        state = self._rich_state()
        results, valid = run_policy(policy, state, n=50)
        valid_set = set(valid)
        for action, _, _ in results:
            assert action in valid_set, f"Invalid action returned: {action}"

    def test_low_temp_output_always_valid(self, policy):
        """Low-temperature (near-greedy) mode must also return valid actions."""
        state = self._rich_state()
        results, valid = run_policy(policy, state, n=10, temperature=0.1)
        valid_set = set(valid)
        for action, _, _ in results:
            assert action in valid_set

    def test_log_prob_requires_grad_training(self, policy):
        """log_prob must carry a grad_fn so REINFORCE can back-propagate."""
        policy.train()
        state = self._rich_state()
        results, _ = run_policy(policy, state, n=20)
        for _, lp, _ in results:
            if lp is None:
                continue  # fallback actions return None — no gradient expected
            assert lp.requires_grad, "log_prob has no grad_fn — backward() will crash"

    def test_log_prob_no_grad_eval(self, policy):
        """Under torch.no_grad() (eval), tensors have no grad_fn — that's expected."""
        policy.eval()
        state = self._rich_state()
        valid = generate_valid_actions(state, include_blocks=False)
        pyg, obj_idx, _ = state_to_pyg(state)
        with torch.no_grad():
            _, lp, _ = policy(pyg, obj_idx, valid, temperature=0.1)
        if lp is not None:
            assert not lp.requires_grad

    def test_entropy_nonneg(self, policy):
        """Entropy of any distribution is >= 0."""
        state = self._rich_state()
        results, _ = run_policy(policy, state, n=20)
        for _, _, ent in results:
            if ent is None:
                continue
            assert ent.item() >= -1e-6, f"Negative entropy: {ent.item()}"

    def test_log_prob_nonpositive(self, policy):
        """Log-probability of any event is <= 0."""
        policy.train()
        state = self._rich_state()
        results, _ = run_policy(policy, state, n=20)
        for _, lp, _ in results:
            if lp is None:
                continue
            assert lp.item() <= 1e-6, f"log_prob > 0: {lp.item()}"


# ══════════════════════════════════════════════════════════════════════════
# ExploitService — canonical host-first decoding
# ══════════════════════════════════════════════════════════════════════════

class TestExploitService:
    """
    Core invariant: target_service must be in state.known_services[target_host].
    This agent now canonicalizes ExploitService to a host-level decision and fills
    target_service deterministically with one valid service for that host.
    """

    def _distinct_services_state(self):
        """Host 1 runs only ssh; host 2 runs only http."""
        return make_state(
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {HTTP},
            },
        )

    def _shared_service_state(self):
        """Both hosts run the same ssh service — deduplication scenario."""
        return make_state(
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {SSH},
            },
        )

    def _mixed_services_state(self):
        """Host 1: ssh only; host 2: ssh + http."""
        return make_state(
            hosts={IP("192.168.1.1"), IP("192.168.1.2")},
            services={
                IP("192.168.1.1"): {SSH},
                IP("192.168.1.2"): {SSH, HTTP},
            },
        )

    def _check_exploit_invariant(self, policy, state, n=60):
        results, _ = run_policy(policy, state, n=n)
        for action, _, _ in results:
            if action.type == ActionType.ExploitService:
                tgt_host = action.parameters["target_host"]
                tgt_svc  = action.parameters["target_service"]
                assert tgt_svc in state.known_services[tgt_host], (
                    f"Service {tgt_svc} is not on host {tgt_host}. "
                    f"Available: {state.known_services[tgt_host]}"
                )

    def test_service_on_host_distinct_services(self, policy):
        """target_service must be on target_host — distinct services per host."""
        self._check_exploit_invariant(policy, self._distinct_services_state())

    def test_service_on_host_shared_service(self, policy):
        """target_service must be on target_host — even when ssh is deduplicated."""
        self._check_exploit_invariant(policy, self._shared_service_state())

    def test_service_on_host_mixed_services(self, policy):
        """target_service must be on target_host — host 1 has ssh, host 2 has ssh+http."""
        self._check_exploit_invariant(policy, self._mixed_services_state())

    def test_exploit_service_actions_are_canonicalized(self):
        """Only one canonical ExploitService action should remain per target host."""
        state = self._mixed_services_state()
        valid = generate_valid_actions(state, include_blocks=False)
        reduced = _canonicalize_valid_actions(valid)
        exploit_actions = [a for a in reduced if a.type == ActionType.ExploitService]
        target_hosts = {a.parameters["target_host"] for a in exploit_actions}
        assert len(exploit_actions) == len(target_hosts)
        for action in exploit_actions:
            tgt_host = action.parameters["target_host"]
            tgt_svc = action.parameters["target_service"]
            assert tgt_svc == min(state.known_services[tgt_host], key=str)


# ══════════════════════════════════════════════════════════════════════════
# ExfiltrateData — four-step correctness
# ══════════════════════════════════════════════════════════════════════════

class TestExfiltrateData:
    """
    Two invariants:
      1. action.data must be in state.known_data[action.source_host]
      2. action.target_host != action.source_host  (can't exfiltrate to yourself)
    Additionally, when there are multiple controlled destination hosts the
    sec_head must score them rather than choosing arbitrarily.
    """

    def _single_data_two_destinations(self):
        """
        One data item on an external host; two controlled hosts as destinations.
        This forces sec_head to choose between two candidate destination nodes.
        """
        return make_state(
            controlled={IP("10.0.0.1"), IP("10.0.0.2")},
            hosts={IP("192.168.1.1")},
            networks={Network("192.168.1.0", 24), Network("10.0.0.0", 24)},
            data={IP("192.168.1.1"): {CREDS}},
        )

    def _two_data_one_destination(self):
        """
        Two data items on one host; one controlled destination.
        This forces tgt_head to score among multiple data nodes.
        """
        return make_state(
            controlled={IP("10.0.0.1")},
            hosts={IP("192.168.1.1")},
            networks={Network("192.168.1.0", 24), Network("10.0.0.0", 24)},
            data={IP("192.168.1.1"): {CREDS, WEBDATA}},
        )

    def _two_data_two_destinations(self):
        """Both data choice and destination choice are non-trivial."""
        return make_state(
            controlled={IP("10.0.0.1"), IP("10.0.0.2")},
            hosts={IP("192.168.1.1")},
            networks={Network("192.168.1.0", 24), Network("10.0.0.0", 24)},
            data={IP("192.168.1.1"): {CREDS, WEBDATA}},
        )

    def _check_exfil_invariants(self, policy, state, n=60):
        results, valid = run_policy(policy, state, n=n)
        valid_set = set(valid)
        for action, _, _ in results:
            if action.type == ActionType.ExfiltrateData:
                src  = action.parameters["source_host"]
                data = action.parameters["data"]
                dst  = action.parameters["target_host"]
                assert action in valid_set, "ExfiltrateData action not in valid_actions"
                assert data in state.known_data[src], (
                    f"Data {data} not on source host {src}"
                )
                assert dst != src, "Exfiltration source and destination are the same host"

    def test_data_on_source_single_data(self, policy):
        self._check_exfil_invariants(policy, self._single_data_two_destinations())

    def test_data_on_source_multiple_data(self, policy):
        self._check_exfil_invariants(policy, self._two_data_one_destination())

    def test_data_on_source_both_multiple(self, policy):
        self._check_exfil_invariants(policy, self._two_data_two_destinations())

    def test_exfiltrate_uses_secondary_head(self):
        """ACTION_SECONDARY_PARAM must map ExfiltrateData → target_host."""
        assert ActionType.ExfiltrateData in ACTION_SECONDARY_PARAM
        assert ACTION_SECONDARY_PARAM[ActionType.ExfiltrateData] == "target_host"
        assert ACTION_SECONDARY_NODE_TYPE[ActionType.ExfiltrateData] == "host"

    def test_both_destinations_reachable(self, policy):
        """
        With two possible destinations, both should appear across stochastic runs.
        This would fail if the destination were hard-coded or always random.choice.
        """
        state = self._single_data_two_destinations()
        results, _ = run_policy(policy, state, n=100)
        destinations = {
            action.parameters["target_host"]
            for action, _, _ in results
            if action.type == ActionType.ExfiltrateData
        }
        # Both destinations must be reachable (policy is stochastic with random init)
        assert len(destinations) == 2, (
            f"Only reached destinations: {destinations}. "
            "sec_head may not be scoring them properly."
        )

    def test_both_data_items_reachable(self, policy):
        """With two data items, both should be chosen across stochastic runs."""
        state = self._two_data_one_destination()
        results, _ = run_policy(policy, state, n=100)
        chosen_data = {
            action.parameters["data"]
            for action, _, _ in results
            if action.type == ActionType.ExfiltrateData
        }
        assert len(chosen_data) == 2, (
            f"Only reached data items: {chosen_data}. "
            "tgt_head may not be scoring multiple data nodes."
        )


# ══════════════════════════════════════════════════════════════════════════
# Simple action types — three-step, no sec_head
# ══════════════════════════════════════════════════════════════════════════

class TestSimpleActionTypes:
    """ScanNetwork, FindServices, FindData use only three steps (sec_head inactive)."""

    def test_scan_network_valid(self, policy):
        """ScanNetwork actions must target a known network."""
        state = make_state(
            networks={Network("192.168.1.0", 24), Network("10.0.0.0", 24)},
        )
        results, valid = run_policy(policy, state, n=20)
        valid_set = set(valid)
        for action, lp, _ in results:
            assert action in valid_set
            if action.type == ActionType.ScanNetwork:
                assert action.parameters["target_network"] in state.known_networks
            if lp is not None:
                assert lp.requires_grad

    def test_find_services_valid(self, policy):
        """FindServices must target a known host."""
        state = make_state(hosts={IP("192.168.1.1"), IP("192.168.1.2")})
        results, valid = run_policy(policy, state, n=20)
        valid_set = set(valid)
        for action, _, _ in results:
            assert action in valid_set
            if action.type == ActionType.FindServices:
                assert action.parameters["target_host"] in state.known_hosts

    def test_find_data_targets_controlled_host(self, policy):
        """FindData: source_host == target_host, both must be controlled."""
        state = make_state(
            controlled={IP("10.0.0.1"), IP("10.0.0.2")},
        )
        results, valid = run_policy(policy, state, n=20)
        valid_set = set(valid)
        for action, _, _ in results:
            assert action in valid_set
            if action.type == ActionType.FindData:
                tgt = action.parameters["target_host"]
                src = action.parameters["source_host"]
                assert tgt == src, "FindData: source and target must be the same host"
                assert tgt in state.controlled_hosts


class TestCLIArgs:
    """Test that wandb CLI args are registered and parse correctly."""

    def _parse(self, args):
        import blackbox_pure_gnn_agent as m
        parser = m._build_arg_parser()
        return parser.parse_args(args)

    def test_wandb_flag_defaults_false(self):
        args = self._parse([])
        assert args.wandb is False

    def test_wandb_flag_can_be_set(self):
        args = self._parse(["--wandb"])
        assert args.wandb is True

    def test_wandb_project_default(self):
        args = self._parse([])
        assert args.wandb_project == "sgrl-sec-blackbox-gnn"

    def test_wandb_project_can_be_set(self):
        args = self._parse(["--wandb_project", "my-project"])
        assert args.wandb_project == "my-project"

    def test_wandb_entity_default_none(self):
        args = self._parse([])
        assert args.wandb_entity is None

    def test_wandb_entity_can_be_set(self):
        args = self._parse(["--wandb_entity", "my-team"])
        assert args.wandb_entity == "my-team"


class TestRollingWinRate:
    def test_empty_deque_returns_zero(self):
        from blackbox_pure_gnn_agent import _rolling_win_rate
        assert _rolling_win_rate(deque()) == 0.0

    def test_all_wins(self):
        from blackbox_pure_gnn_agent import _rolling_win_rate
        d = deque([1, 1, 1, 1], maxlen=100)
        assert _rolling_win_rate(d) == 1.0

    def test_half_wins(self):
        from blackbox_pure_gnn_agent import _rolling_win_rate
        d = deque([1, 0, 1, 0], maxlen=100)
        assert _rolling_win_rate(d) == 0.5

    def test_maxlen_respected(self):
        from blackbox_pure_gnn_agent import _rolling_win_rate
        d = deque(maxlen=3)
        for _ in range(10):
            d.append(0)
        d.append(1)       # deque is now [0, 0, 1]
        assert _rolling_win_rate(d) == pytest.approx(1/3)
