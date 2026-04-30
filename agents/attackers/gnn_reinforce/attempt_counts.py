"""Per-episode per-action attempt counters.

Tracks how many times each (action_type, target) pair has been attempted in
the current episode. The counts are encoded into GNN node features by
state_to_pyg, letting the policy see "I already tried this" and break out
of Markov-identical stuck loops.

The container is owned by the agent/trainer, not GameState, because the
counts are a property of the agent's interaction history, not the environment.
"""
from dataclasses import dataclass, field
from typing import Dict

from netsecgame.game_components import Action, ActionType, IP, Network, Data


@dataclass
class AttemptCounts:
    scan: Dict[Network, int] = field(default_factory=dict)
    findservices: Dict[IP, int] = field(default_factory=dict)
    finddata: Dict[IP, int] = field(default_factory=dict)
    exploit: Dict[IP, int] = field(default_factory=dict)
    exfil: Dict[Data, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.scan.clear()
        self.findservices.clear()
        self.finddata.clear()
        self.exploit.clear()
        self.exfil.clear()

    def record(self, action: Action) -> None:
        params = action.parameters
        t = action.type
        if t == ActionType.ScanNetwork:
            key = params.get("target_network")
            if key is not None:
                self.scan[key] = self.scan.get(key, 0) + 1
        elif t == ActionType.FindServices:
            key = params.get("target_host")
            if key is not None:
                self.findservices[key] = self.findservices.get(key, 0) + 1
        elif t == ActionType.FindData:
            key = params.get("target_host")
            if key is not None:
                self.finddata[key] = self.finddata.get(key, 0) + 1
        elif t == ActionType.ExploitService:
            key = params.get("target_host")
            if key is not None:
                self.exploit[key] = self.exploit.get(key, 0) + 1
        elif t == ActionType.ExfiltrateData:
            key = params.get("data")
            if key is not None:
                self.exfil[key] = self.exfil.get(key, 0) + 1
