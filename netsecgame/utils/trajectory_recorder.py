import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from netsecgame.game_components import Action, GameState
from netsecgame.utils.utils import store_trajectories_to_jsonl


class TrajectoryRecorder:
    def __init__(self, agent_name: str, agent_role: str) -> None:
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.logger = logging.getLogger(f"TrajectoryRecorder-{agent_name}")
        self._data: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self._data = {
            "trajectory": {
                "states": [],
                "actions": [],
                "rewards": [],
            },
            "end_reason": None,
            "agent_role": self.agent_role,
            "agent_name": self.agent_name,
        }

    def add_step(
        self,
        action: Action,
        reward: float,
        next_state: GameState,
        end_reason: Optional[str] = None,
    ) -> None:
        if len(self._data["trajectory"]["states"]) == 0:
            self.logger.warning(
                "The initial state has not been recorded yet. Call add_initial_state() first."
            )
        self._data["trajectory"]["actions"].append(action.as_dict)
        self._data["trajectory"]["rewards"].append(reward)
        self._data["trajectory"]["states"].append(next_state.as_dict)

        if end_reason:
            self._data["end_reason"] = end_reason

    def add_initial_state(self, state: GameState) -> None:
        self._data["trajectory"]["states"].append(state.as_dict)

    def get_trajectory(self) -> Dict[str, Any]:
        return self._data

    def save_to_file(self, location: str = "./logs/trajectories", filename: str = None) -> None:
        if filename is None:
            filename = f"{datetime.now():%Y-%m-%d}_{self.agent_name}_{self.agent_role}"
        try:
            store_trajectories_to_jsonl(self._data, location, filename)
            self.logger.debug(f"Trajectory stored in {os.path.join(location, filename)}.jsonl")
        except Exception as exc:
            self.logger.error(f"Failed to store trajectory: {exc}")
