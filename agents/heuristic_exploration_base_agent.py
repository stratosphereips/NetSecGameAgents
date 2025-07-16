# Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
from AIDojoCoordinator.game_components import Action, Observation
from NetSecGameAgents.agents.base_agent import BaseAgent
from NetSecGameAgents.agents.agent_utils import heuristic_network_expansion

class HeuristicExplorationBaseAgent(BaseAgent):
    """
    This agent extends the BaseAgent to perform heuristic extension of known networks.
    It uses a heuristic approach to expand the known networks based on the current state.
    The agent can be configured to explore known hosts or not, and it uses a network offset
    to determine how far to expand the known networks.

    The `make_step` method is overridden to include the heuristic network expansion
    after the base step is made. This allows the agent to update its known networks
    based on the actions taken and the observations received. This enables the agent to scan previously unknown networks
    and expand its knowledge base, which is crucial for effective exploration in some network security environments.
    """
    def __init__(self, host, port, role:str, net_offset=1, explore_known_hosts=False)->None:
        super().__init__(host, port, role)
        self._offset = net_offset
        self._explore_known_hosts = explore_known_hosts
    
    def make_step(self, action: Action)->Observation:
        """
        Executes a step in the environment using the provided action.

        Args:
            action (Action): The action to perform in the environment.

        Returns:
            Observation: The observation resulting from the action, with known networks
            expanded heuristically.
        """
        observation = super().make_step(action)
        
        if observation is None:
            self._logger.error("Failed to make step, received None observation.")
            return None
        else:
            # expand the known networks heuristically
            self._logger.debug(f"Expanding known network heuristically: {observation.state.known_networks}")
            new_state = heuristic_network_expansion(observation.state, self._offset, self._explore_known_hosts)
            self._logger.debug(f"\tResult: {new_state.known_networks}")
            observation = Observation(new_state, observation.reward, observation.end, observation.info)
        return observation