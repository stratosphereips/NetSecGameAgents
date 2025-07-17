import argparse
import numpy as np
import json
from os import path
from AIDojoCoordinator.game_components import Action, Observation, GameState
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import generate_valid_actions


class ActionListAgent(BaseAgent):
    """
    Extension of the BaseAgent that provides a list of all possible actions
    and allows the agent to query actions by index.
    This agent registers with the game environment and retrieves the action list.

    Compatible with the WhiteBoxNSGCoordinator.
    """

    def __init__(self, host, port, role: str):
        super().__init__(host, port, role)
        self._action_list = []
        self._action_to_idx = {}

    def register(self) -> Observation:
        """
        Register the agent with the game environment. Parse the action list in the response.
        """
        obs = super().register()
        if isinstance(obs.info, dict) and 'all_actions' in obs.info.keys():
            data = json.loads(obs.info['all_actions'])
            self._action_list = [Action.from_dict(action_dict) for action_dict in data]
            self._action_to_idx = {action:idx for idx, action in enumerate(self._action_list)}
        else:
            raise KeyError("Expected key 'all_actions' in the Observation info after registration.")
        return obs
    
    def action_space(self) -> list:
        """
        Return the list of all actions available to the agent.
        """
        return self._action_list

    def get_action_index(self, action: Action) -> int:
        """
        Get the index of an action in the action list.
        """
        return self._action_to_idx.get(action, -1)
    
    def get_action(self, action_index: int) -> Action:
        """
        Get the action by its index in the action list.
        """
        if 0 <= action_index < len(self._action_list):
            return self._action_list[action_index]
        else:
            raise IndexError("Action index out of range.")
    
    def get_valid_action_mask(self , state: GameState) -> np.ndarray:
        """
        Get the action vector, which is a list of all actions.
        """
        mask = np.zeros(len(self._action_list), dtype=bool)
        for valid_action in generate_valid_actions(state, include_blocks=False):
            if valid_action in self._action_to_idx:
                mask[self._action_to_idx[valid_action]] = True
        return mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    
    args = parser.parse_args()
    log_filename = path.dirname(path.abspath(__file__)) + '/action_List_base_agent.log'
    agent = ActionListAgent(args.host, args.port, "Attacker")
    observation = agent.register()
    print(f"Total actions: {len(agent._action_list)}")
    print(f"Valid action mask: {agent.get_valid_action_mask(observation.state)}")